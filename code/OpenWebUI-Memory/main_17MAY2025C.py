async def get_past_interactions(user_input, context="general", use_cache=True):
    log.info(f"Retrieving past interactions for input: {user_input[:100]}... with context: {context}")
    if use_cache and hasattr(request.state, 'memory_cache') and context in request.state.memory_cache:
        log.info(f"Using cached memories for context: {context}")
        return request.state.memory_cache[context]

    try:
        client = await chromadb.AsyncHttpClient(host=chroma_host, port=chroma_port)
        collection = await client.get_or_create_collection(
            name="chat_memories",
            metadata={"hnsw:M": 32, "hnsw:ef_construction": 400}
        )
        embeddings = embed_text([user_input])
        if embeddings is None or not embeddings:
            log.error("Failed to generate text embeddings")
            return "No relevant past interactions found."
        query_embedding = embeddings[0]
        log.info(f"Query embedding: {query_embedding[:10]}...")  # Log the first 10 elements
        results = await collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"context": context}
        )
        past_interactions = ""
        if results["documents"]:
            log.info(f"Retrieved {len(results['documents'][0])} past interactions")
            ids_to_update = results["ids"][0][:5]
            metadatas_to_update = []
            for meta in results["metadatas"][0][:5]:
                access_count = int(meta.get('access_count', 0)) + 1
                meta['access_count'] = access_count
                metadatas_to_update.append(meta)
            try:
                await collection.update(
                    ids=ids_to_update,
                    metadatas=metadatas_to_update
                )
                log.info(f"Updated access_count for {len(ids_to_update)} documents")
            except Exception as e:
                log.error(f"Failed to update access_count: {e}")

            combined_scores = []
            for i, (distance, meta) in enumerate(zip(results["distances"][0][:5], results["metadatas"][0][:5])):
                access_count = int(meta.get('access_count', 0))
                normalized_access_count = min(access_count, 100) / 100.0
                similarity_score = 1.0 - distance
                combined_score = 0.9 * similarity_score + 0.1 * normalized_access_count
                combined_scores.append((i, combined_score))
            combined_scores.sort(key=lambda x: x[1], reverse=True)

            for i, _ in combined_scores:
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                timestamp = float(meta.get('timestamp', 0))
                time_str = datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
                raw_image_base64 = meta.get('raw_image_base64', 'N/A')[:50] + "..." if meta.get('raw_image_base64') else 'N/A'
                access_count = int(meta.get('access_count', 0))
                past_interactions += f"Past user [{time_str}]: {doc}\nLyra: {meta['response']}\nRaw Image Base64: {raw_image_base64}\nAccess Count: {access_count}\n\n"
        else:
            log.info("No past interactions found")
            past_interactions = "No relevant past interactions found."
        if not hasattr(request.state, 'memory_cache'):
            request.state.memory_cache = {}
        request.state.memory_cache[context] = past_interactions
        return past_interactions
    except Exception as e:
        log.error(f"ChromaDB retrieval failed: {e}")
        return "ChromaDB unavailable, relying on current context."