async def collect_agent_state_messages(agent, user_input: str):
    final_messages = None

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
        version="v2",
    ):
        if chunk["type"] == "values":
            final_messages = chunk["data"]["messages"]

    return final_messages or []
	
	-----------------------------------------------
	
from langchain_core.messages import convert_to_messages
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.utils import message_chunk_to_message


async def collect_agent_messages(agent, user_input: str):
    input_messages = convert_to_messages([
        {"role": "user", "content": user_input}
    ])

    full_ai_chunk = None

    async for chunk in agent.astream(
        {"messages": input_messages},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] != "messages":
            continue

        token, metadata = chunk["data"]

        if not isinstance(token, AIMessageChunk):
            continue

        full_ai_chunk = token if full_ai_chunk is None else full_ai_chunk + token

    if full_ai_chunk is None:
        return input_messages

    ai_message = message_chunk_to_message(full_ai_chunk)

    return input_messages + [ai_message]
	
	
	
	
	
	
	---------------------------
	
	
	
	
	
	
from langchain_core.messages import convert_to_messages

async def collect_final_messages(agent, user_text: str):
    final_state = None

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_text}]},
        stream_mode="values",
        version="v2",
    ):
        if chunk["type"] == "values":
            final_state = chunk["data"]

    if not final_state:
        return []

    return convert_to_messages(final_state["messages"])
	
	
	-------------------------------
	
	
	
	from langchain_core.messages import convert_to_messages
from langchain_core.messages.ai import AIMessageChunk

async def stream_agent(agent, user_text: str):
    final_state = None

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_text}]},
        stream_mode=["messages", "values"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]

            if isinstance(token, AIMessageChunk) and token.text:
                yield {
                    "type": "token",
                    "text": token.text,
                    "node": metadata.get("langgraph_node"),
                }

        elif chunk["type"] == "values":
            final_state = chunk["data"]

    final_messages = (
        convert_to_messages(final_state["messages"])
        if final_state and "messages" in final_state
        else []
    )

    yield {
        "type": "final_messages",
        "messages": final_messages,
    }
