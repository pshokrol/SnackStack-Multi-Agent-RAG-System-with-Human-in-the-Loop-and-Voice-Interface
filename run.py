import os
import re
import io
import uuid
import logging
import numpy as np
from dotenv import load_dotenv
from typing import TypedDict, Optional, Annotated

# ── Load API key ────────────────────────────────────────────────────────────
load_dotenv()

# ── config ──────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model='gpt-4o', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

def setup_logger(name: str = 'snackstack') -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# ── data/menu.py ────────────────────────────────────────────────────────────
MENU_CATALOG = [
    {
        'id': 'ITEM-001', 'name': 'Margherita Pizza', 'category': 'Main',
        'cuisine': 'Italian', 'price': 29, 'rating': 4.7,
        'dietary_tags': ['Veg'],
        'description': 'Classic thin crust with tomato, mozzarella, basil',
        'availability': True
    },
    {
        'id': 'ITEM-002', 'name': 'Vegan Pasta Primavera', 'category': 'Main',
        'cuisine': 'Italian', 'price': 34, 'rating': 4.5,
        'dietary_tags': ['Vegan'],
        'description': 'Penne with seasonal vegetables, olive oil, garlic',
        'availability': True
    },
    {
        'id': 'ITEM-003', 'name': 'Joojeh Kabob', 'category': 'Main',
        'cuisine': 'Persian', 'price': 37, 'rating': 4.9,
        'dietary_tags': ['GF'],
        'description': 'Tender saffron-marinated chicken skewered and grilled over open flames',
        'availability': True
    },
    {
        'id': 'ITEM-004', 'name': 'Vegan Buddha Bowl', 'category': 'Main',
        'cuisine': 'Fusion', 'price': 31, 'rating': 4.6,
        'dietary_tags': ['Vegan', 'GF'],
        'description': 'Quinoa, chickpeas, avocado, greens, tahini',
        'availability': True
    },
    {
        'id': 'ITEM-005', 'name': 'Classic Cheeseburger', 'category': 'Main',
        'cuisine': 'American', 'price': 25, 'rating': 4.4,
        'dietary_tags': [],
        'description': 'Beef patty, cheddar, lettuce, tomato, brioche bun',
        'availability': True
    },
    {
        'id': 'ITEM-006', 'name': 'Ghormeh Sabzi', 'category': 'Main',
        'cuisine': 'Persian', 'price': 19, 'rating': 4.8,
        'dietary_tags': ['Veg', 'GF'],
        'description': 'Slow cooking tender chunks of beef or lamb red kidney beans and a mountain of finely chopped sautéed fresh herbs',
        'availability': True
    },
    {
        'id': 'ITEM-007', 'name': 'Aglio e Olio', 'category': 'Main',
        'cuisine': 'Italian', 'price': 27, 'rating': 4.5,
        'dietary_tags': ['Vegan'],
        'description': 'Spagwhat hetti with garlic, chilli, olive oil, parsley',
        'availability': True
    },
    {
        'id': 'ITEM-008', 'name': 'Faloodeh Shirazi', 'category': 'Beverage',
        'cuisine': 'Persian', 'price': 10, 'rating': 4.7,
        'dietary_tags': ['Veg', 'GF'],
        'description': 'Semifrozen starch noodles tossed in a sweet syrup flavored with rosewater and a splash of tart lime juice',
        'availability': True
    },
]

# ── data/orders.py ──────────────────────────────────────────────────────────
ORDER_DB = {
    'ORD-201': {
        'order_id': 'ORD-201', 'item_id': 'ITEM-003',
        'item_name': 'Butter Chicken', 'customer_name': 'Priya Nair',
        'customer_email': 'priya@example.com', 'status': 'Out for Delivery',
        'price': 37, 'order_date': '2025-05-28',
        'estimated_delivery': '2025-05-28', 'tracking_id': 'SS201TRK'
    },
    'ORD-202': {
        'order_id': 'ORD-202', 'item_id': 'ITEM-001',
        'item_name': 'Margherita Pizza', 'customer_name': 'Arjun Mehta',
        'customer_email': 'arjun@example.com', 'status': 'Placed',
        'price': 29, 'order_date': '2025-05-28',
        'estimated_delivery': '2025-05-28', 'tracking_id': 'SS202TRK'
    },
    'ORD-203': {
        'order_id': 'ORD-203', 'item_id': 'ITEM-005',
        'item_name': 'Classic Cheeseburger', 'customer_name': 'Sneha Roy',
        'customer_email': 'sneha@example.com', 'status': 'Preparing',
        'price': 25, 'order_date': '2025-05-28',
        'estimated_delivery': '2025-05-28', 'tracking_id': 'SS203TRK'
    },
    'ORD-204': {
        'order_id': 'ORD-204', 'item_id': 'ITEM-004',
        'item_name': 'Vegan Buddha Bowl', 'customer_name': 'Rahul Das',
        'customer_email': 'rahul@example.com', 'status': 'Delivered',
        'price': 31, 'order_date': '2025-05-27',
        'estimated_delivery': '2025-05-27', 'tracking_id': 'SS204TRK'
    },
    'ORD-205': {
        'order_id': 'ORD-205', 'item_id': 'ITEM-006',
        'item_name': 'Paneer Tikka', 'customer_name': 'Kavya Sharma',
        'customer_email': 'kavya@example.com', 'status': 'Placed',
        'price': 19, 'order_date': '2025-05-28',
        'estimated_delivery': '2025-05-28', 'tracking_id': 'SS205TRK'
    },
}

# ── tools/rag.py ────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

def _build_menu_docs():
    """Convert menu catalog entries into LangChain Documents."""
    docs = []
    for dish in MENU_CATALOG:
        tags = ', '.join(dish['dietary_tags']) if dish['dietary_tags'] else 'None'
        content = (
            f"Name: {dish['name']}\n"
            f"Cuisine: {dish['cuisine']}\n"
            f"Price: ${dish['price']} USD\n"
            f"Rating: {dish['rating']}/5\n"
            f"Dietary: {tags}\n"
            f"Description: {dish['description']}\n"
            f"Available: {'Yes' if dish['availability'] else 'No'}"
        )
        docs.append(Document(page_content=content, metadata={'id': dish['id'], 'name': dish['name']}))
    return docs

_menu_docs = _build_menu_docs()
_vectorstore = Chroma.from_documents(_menu_docs, embeddings, collection_name='snackstack_menu')
_retriever = _vectorstore.as_retriever(search_kwargs={'k': 3})

@tool
def search_menu_catalog(query: str) -> str:
    """Search the SnackStack menu catalog using semantic similarity.
    Use this for any question about food, dishes, cuisine, price,
    dietary preferences, or menu items."""
    results = _retriever.invoke(query)
    if not results:
        return 'No matching dishes found in the menu.'
    return '\n\n---\n\n'.join(doc.page_content for doc in results)

@tool
def get_order_status(identifier: str) -> str:
    """Look up an order by Order ID (e.g. ORD-201), Tracking ID (e.g. SS201TRK),
    or customer email. Returns full order details or an error message."""
    identifier = identifier.strip()
    if identifier.upper() in ORDER_DB:
        order = ORDER_DB[identifier.upper()]
    else:
        order = next(
            (o for o in ORDER_DB.values()
             if o['tracking_id'].upper() == identifier.upper()
             or o['customer_email'].lower() == identifier.lower()),
            None
        )
    if not order:
        return f'No order found for identifier: {identifier}'
    return (
        f"Order ID: {order['order_id']}\n"
        f"Item: {order['item_name']} (${order['price']})\n"
        f"Customer: {order['customer_name']} ({order['customer_email']})\n"
        f"Status: {order['status']}\n"
        f"Tracking: {order['tracking_id']}\n"
        f"Order Date: {order['order_date']}\n"
        f"Est. Delivery: {order['estimated_delivery']}"
    )

# ── state.py ────────────────────────────────────────────────────────────────
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class StackState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    route: list[str]
    menu_response: Optional[str]
    order_response: Optional[str]
    final_answer: Optional[str]

# ── agents/prompts.py ────────────────────────────────────────────────────────
ORCHESTRATOR_SYSTEM = """\
You are the SnackStack routing orchestrator. Classify the user query and decide
which specialist agent(s) should handle it.

Routing rules:
- 'menu_agent'  : food/menu questions, cuisine, dietary needs, prices, ratings,
                  greetings, small talk, unclear intent
- 'order_agent' : order tracking, order status, delivery updates
- both          : query clearly covers both menu AND order topics

Default to ['menu_agent'] when unsure.
NEVER route greetings to 'order_agent'.
"""

MENU_AGENT_SYSTEM = """\
You are the SnackStack Menu Agent. You help customers explore the food menu.

You have access to the search_menu_catalog tool. Use it to find dishes that
match the user's request. Always call the tool for food/menu queries.
For greetings, respond warmly without calling any tool.

Format responses clearly: include dish name, price ($), dietary tags, and
a brief description. Be friendly and conversational.
"""

ORDER_AGENT_SYSTEM = """\
You are the SnackStack Order Agent. You look up order status for customers.

You have access to the get_order_status tool. Call it with the identifier
(Order ID like ORD-201, Tracking ID like SS201TRK, or customer email).

If the user has not provided any identifier, do NOT guess — indicate that
you need more information.
Present order results in a clear, friendly format.
"""

SYNTHESIZER_SYSTEM = """\
You are the SnackStack response synthesizer. You receive outputs from one or
more specialist agents and produce a single, coherent, friendly reply.

- If both menu and order responses are present, weave them naturally.
- If only one response is present, clean it up and present it directly.
- Keep the tone warm, helpful, and concise.
"""

# ── agents/orchestrator.py ───────────────────────────────────────────────────
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.types import Command, interrupt
from typing import Literal

class RouteDecision(BaseModel):
    agents: list[Literal['menu_agent', 'order_agent']] = Field(
        description='Which agent(s) should handle this query.'
    )
    reasoning: str = Field(description='One-sentence explanation of the routing decision.')

_routing_llm = llm.with_structured_output(RouteDecision)

def orchestrator_node(state: StackState) -> Command:
    logger.info('Orchestrator: routing query: %s', state['user_query'][:80])
    decision: RouteDecision = _routing_llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=state['user_query'])
    ])
    logger.info('Orchestrator: → %s | reason: %s', decision.agents, decision.reasoning)
    next_node = decision.agents[0]
    return Command(
        update={'route': decision.agents, 'menu_response': None, 'order_response': None},
        goto=next_node
    )

# ── agents/menu_agent.py ─────────────────────────────────────────────────────
MAX_TOOL_ITERATIONS = 5
_menu_llm = llm.bind_tools([search_menu_catalog])

def menu_agent_node(state: StackState) -> Command:
    logger.info('Menu Agent: processing query')
    messages = [
        SystemMessage(content=MENU_AGENT_SYSTEM),
        HumanMessage(content=state['user_query'])
    ]
    for _ in range(MAX_TOOL_ITERATIONS):
        response: AIMessage = _menu_llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            if tc['name'] == 'search_menu_catalog':
                result = search_menu_catalog.invoke(tc['args'])
            else:
                result = f'Unknown tool: {tc["name"]}'
            messages.append(ToolMessage(content=result, tool_call_id=tc['id']))
    else:
        response = AIMessage(content='I reached my search limit. Here is what I found so far.')

    logger.info('Menu Agent: done')
    route = state.get('route', ['menu_agent'])
    goto = 'order_agent' if 'order_agent' in route else 'synthesizer'
    return Command(update={'menu_response': response.content}, goto=goto)

# ── agents/order_agent.py ────────────────────────────────────────────────────
_order_llm = llm.bind_tools([get_order_status])
_ORDER_ID_RE = re.compile(r'\bORD-\d+\b', re.IGNORECASE)
_TRACKING_RE = re.compile(r'\bSS\d+TRK\b', re.IGNORECASE)
_EMAIL_RE    = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')

def _extract_identifier(text: str) -> str | None:
    for pattern in [_ORDER_ID_RE, _TRACKING_RE, _EMAIL_RE]:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None

def order_agent_node(state: StackState) -> Command:
    logger.info('Order Agent: processing query')
    query = state['user_query']
    identifier = _extract_identifier(query)
    if not identifier:
        logger.info('Order Agent: no identifier found, triggering HITL interrupt')
        user_response = interrupt(
            '🔍 I need your order details to look it up. '
            'Please provide your Order ID (e.g. ORD-201), '
            'Tracking ID (e.g. SS201TRK), or email address:'
        )
        query = user_response
        identifier = _extract_identifier(query) or query.strip()

    messages = [
        SystemMessage(content=ORDER_AGENT_SYSTEM),
        HumanMessage(content=f'Look up this order: {identifier}')
    ]
    for _ in range(MAX_TOOL_ITERATIONS):
        response: AIMessage = _order_llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            if tc['name'] == 'get_order_status':
                result = get_order_status.invoke(tc['args'])
            else:
                result = f'Unknown tool: {tc["name"]}'
            messages.append(ToolMessage(content=result, tool_call_id=tc['id']))
    else:
        response = AIMessage(content='I could not retrieve the order details after several attempts.')

    logger.info('Order Agent: done')
    return Command(update={'order_response': response.content}, goto='synthesizer')

# ── agents/synthesizer.py ────────────────────────────────────────────────────
def synthesizer_node(state: StackState) -> Command:
    logger.info('Synthesizer: merging responses')
    parts = []
    if state.get('menu_response'):
        parts.append(f'[Menu Agent]\n{state["menu_response"]}')
    if state.get('order_response'):
        parts.append(f'[Order Agent]\n{state["order_response"]}')

    if not parts:
        return Command(update={'final_answer': 'I could not find any information for your query.'}, goto='__end__')

    combined = '\n\n'.join(parts)
    response: AIMessage = llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=f'Original query: {state["user_query"]}\n\nAgent outputs:\n{combined}')
    ])
    logger.info('Synthesizer: done')
    return Command(update={'final_answer': response.content}, goto='__end__')

# ── graph.py ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(StackState)
builder.add_node('orchestrator', orchestrator_node)
builder.add_node('menu_agent',   menu_agent_node)
builder.add_node('order_agent',  order_agent_node)
builder.add_node('synthesizer',  synthesizer_node)
builder.add_edge(START, 'orchestrator')

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
print('LangGraph compiled successfully')

# ── main.py ──────────────────────────────────────────────────────────────────
class SnackStackAssistant:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        print(f'SnackStack started | thread: {self.thread_id}')

    def _config(self):
        return {'configurable': {'thread_id': self.thread_id}}

    def ask(self, user_input: str) -> str:
        config = self._config()
        state_snapshot = graph.get_state(config)
        if state_snapshot.next:
            result = graph.invoke(Command(resume=user_input), config=config)
        else:
            initial_state: StackState = {
                'messages': [],
                'user_query': user_input,
                'route': [],
                'menu_response': None,
                'order_response': None,
                'final_answer': None,
            }
            result = graph.invoke(initial_state, config=config)

        state_snapshot = graph.get_state(config)
        if state_snapshot.next:
            interrupt_data = state_snapshot.tasks[0].interrupts[0].value
            return f'❓ {interrupt_data}'

        return result.get('final_answer', '(No response generated)')

    def reset(self):
        self.thread_id = str(uuid.uuid4())
        print(f'Conversation reset | new thread: {self.thread_id}')


# ── voice/recorder.py — Speech-to-Text (Whisper) ──────────────────────────
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

_openai_client = OpenAI()

def record_audio(duration: int = 5, sample_rate: int = 16000) -> io.BytesIO:
    """Record audio from microphone and return as WAV bytes."""
    print(f' Recording for {duration} seconds... Speak now!')
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    print('Recording complete')

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    wav_buffer.seek(0)
    return wav_buffer

def transcribe_audio(wav_buffer: io.BytesIO) -> str:
    """Transcribe WAV audio using OpenAI Whisper."""
    wav_buffer.name = 'audio.wav'
    result = _openai_client.audio.transcriptions.create(
        model='whisper-1',
        file=wav_buffer
    )
    return result.text

def listen(duration: int = 5) -> str:
    """Record microphone input and return transcribed text."""
    wav_buffer = record_audio(duration=duration)
    text = transcribe_audio(wav_buffer)
    print(f'You said: {text}')
    return text


# ── voice/speaker.py — Text-to-Speech ─────────────────────────────────────
def speak(text: str, voice: str = 'alloy') -> None:
    """Convert text to speech and play through speakers."""
    print(f'Speaking: {text[:60]}...' if len(text) > 60 else f'Speaking: {text}')

    response = _openai_client.audio.speech.create(
        model='tts-1',
        voice=voice,        # alloy, echo, fable, onyx, nova, shimmer
        input=text,
        response_format='wav'
    )

    wav_bytes = io.BytesIO(response.content)
    audio_data, sample_rate = sf.read(wav_bytes, dtype='int16')
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()


# ── REPLs ─────────────────────────────────────────────────────────────────
def run_text_loop():
    """Text-only interactive loop."""
    assistant = SnackStackAssistant()
    print('\n SnackStack — Text Mode')
    print('Commands: "reset" (new conversation) | "quit" (exit)\n')

    while True:
        try:
            user_input = input('You: ').strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print('Thanks for using SnackStack!')
            break
        if user_input.lower() == 'reset':
            assistant.reset()
            print('Fresh conversation started.\n')
            continue
        response = assistant.ask(user_input)
        print(f'SnackStack: {response}\n')


def run_voice_loop(record_duration: int = 7):
    """Full two-way voice loop — mic in, speaker out."""
    assistant = SnackStackAssistant()
    speak('SnackStack is ready.')
    print('\n Voice Mode | Press Ctrl+C to exit\n')

    while True:
        try:
            input('Press ENTER when ready to speak...')
            user_text = listen(duration=record_duration)

            # Show what was heard and confirm before sending
            if not user_text.strip():
                print('Nothing detected, try again.')
                speak('I did not catch that. Please try again.')
                continue

            print(f'Heard: "{user_text}"')
            confirm = input('Send this? (y/n): ').strip().lower()
            if confirm != 'y':
                print('Discarded, try again.\n')
                continue

            if 'quit' in user_text.lower():
                speak('Goodbye!')
                break
            if 'reset' in user_text.lower():
                assistant.reset()
                speak('Conversation reset.')
                continue

            response = assistant.ask(user_text)

            if response.startswith('?'):
                question = response[2:].strip()
                speak(question)
                followup = listen(duration=record_duration)
                response = assistant.ask(followup)

            speak(response)

        except KeyboardInterrupt:
            speak('Goodbye!')
            break


def run_voice_out_loop():
    """Type queries, hear responses spoken aloud."""
    assistant = SnackStackAssistant()
    print('\n Voice-Out Mode | Type queries, hear responses | "quit" to exit\n')

    while True:
        try:
            user_input = input('You: ').strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            speak('Goodbye!')
            break
        if user_input.lower() == 'reset':
            assistant.reset()
            print(' Reset\n')
            continue
        response = assistant.ask(user_input)
        print(f'SnackStack: {response}')
        speak(response)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n Welcome to SnackStack!')
    print('Select mode:')
    print('  1 - Text chat')
    print('  2 - Full voice (mic + speaker)')
    print('  3 - Type in, speak out')
    choice = input('Enter 1, 2 or 3: ').strip()

    if choice == '2':
        run_voice_loop()
    elif choice == '3':
        run_voice_out_loop()
    else:
        run_text_loop()