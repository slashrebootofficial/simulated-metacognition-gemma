def store_image_with_desc(image_path, desc=None, collection):
    if desc is None:
        desc = os.path.splitext(os.path.basename(image_path))[0]  # e.g., "candy_with_turtle"
    image_emb = embed_image(image_path)
    text_emb = embedding_model.encode(desc)
    collection.add(
        embeddings=[image_emb],
        metadatas=[{"path": image_path, "desc": desc, "desc_embedding": text_emb, "timestamp": datetime.now().isoformat()}]
    )