import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER_URL = "http://127.0.0.1:8100/mcp"

async def main():
    async with streamablehttp_client(SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            print("Tools:", [t.name for t in tools_resp.tools])

            if any(t.name == "lookup_regulation" for t in tools_resp.tools):
                call = await session.call_tool("lookup_regulation", {"regulation_code": "AAMI TIR50"})

                texts = [part.text for part in call.content if hasattr(part, "text")]
                print("lookup_regulation(AAMI TIR50) ->", " ".join(texts) if texts else call.model_dump())

if __name__ == "__main__":
    asyncio.run(main())