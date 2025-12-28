from lyra_memory import LyraMemory
import asyncio

async def test():
    lyra = LyraMemory()
    memories = await lyra.recall_memories('nebula heart')
    print(memories)

asyncio.run(test())
