import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER_URL = "http://127.0.0.1:8000/mcp"

async def main():
    async with streamablehttp_client(SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            print("Tools:", [t.name for t in tools_resp.tools])

            if any(t.name == "add" for t in tools_resp.tools):
                call = await session.call_tool("add", {"a": 2, "b": 3})

                texts = [part.text for part in call.content if hasattr(part, "text")]
                print("add(2,3) ->", " ".join(texts) if texts else call.model_dump())

if __name__ == "__main__":
    asyncio.run(main())