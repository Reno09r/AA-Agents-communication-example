import os
import click
import uvicorn
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from a2a.types import (
    AgentCard, AgentCapabilities, AgentSkill, TaskState, TextPart
)
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

try:
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    print("Document index loaded successfully")
except Exception as e:
    print(f"Warning: Could not load document index: {e}")
    query_engine = None

COMPANY_DATA = """
Our company 'AI Solutions Corp' specialises in developing advanced AI solutions for businesses.

The main products are:
1. Document Analyzer - Automated document and invoice processing system
   - Uses OCR and NLP for data extraction
   - Integrates with ERP systems
   - 99.7% recognition accuracy
   - Price: from $5,000/month for basic version
   - Supports formats: PDF, JPG, PNG, TIFF
   - Processing time: up to 1000 documents per hour

2. Vision AI - warehouse monitoring and inventory solution
   - Computer vision for tracking goods
   - Automatic inventory update
   - Anomaly and shortage detection
   - Price: from $8,000/month
   - Camera support: IP cameras, USB cameras
   - Detection accuracy: 98.5%

Founded in 2020, over 500 successful implementations, offices in 15 countries.
24/7 technical support, 30 days money back guarantee.
"""

async def search_documents(query: str) -> str:
    """Полезна для ответов на вопросы о документах компании."""
    if query_engine:
        try:
            response = await query_engine.aquery(query)
            return str(response)
        except Exception as e:
            return f"Ошибка поиска в документах: {e}"
    return "Система поиска в документах недоступна."

def calculate(expression: str) -> str:
    """Выполняет математические вычисления."""
    try:
        allowed_names = {
            k: v for k, v in __builtins__.items() if k in ['abs', 'min', 'max', 'round', 'sum']
        }
        allowed_names.update({'__builtins__': {}})
        
        result = eval(expression, allowed_names)
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка вычисления: {e}"

expert_agent = FunctionAgent(
    tools=[search_documents, calculate],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt=f"""You are an expert at AI Solutions Corp. You know everything about our products and services.

Company information:
{COMPANY_DATA}

You can:
1. Answer questions about the company's products
2. Perform mathematical calculations
3. Search for information in company documents

Answer in a professional and detailed manner. Always provide specific details about products, including prices and specifications.""",
)

class CompanyExpertExecutor(AgentExecutor):
    """A2A Executor для агента-эксперта компании"""
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Выполняет задачи эксперта компании"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            if event_queue.is_closed():
                print("Warning: Event queue is closed, reopening...")
                event_queue = EventQueue()
                task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            
            task_updater.submit()
            task_updater.start_work()
            
            user_message = context.message.parts[0].root.text
            print(f"Эксперт получил вопрос: {user_message}")
            
            try:
                response = await expert_agent.run(user_message)
                final_text_response = str(response)
            except Exception as agent_error:
                print(f"Ошибка агента LlamaIndex: {agent_error}")
                final_text_response = f"Получен вопрос: {user_message}\n\n{COMPANY_DATA}"
            
            print(f"Эксперт отвечает: {final_text_response[:200]}...")
            
            task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=final_text_response)]
                ),
            )
            
        except Exception as e:
            print(f"Ошибка в CompanyExpertExecutor: {e}")
            try:
                task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
                task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text=f"Ошибка при обработке запроса: {str(e)}")]
                    ),
                )
            except Exception as update_error:
                print(f"Критическая ошибка при обновлении статуса: {update_error}")
        
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Отменяет выполнение задачи"""
        try:
            task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            task_updater.update_status(TaskState.canceled)
        except Exception as e:
            print(f"Ошибка при отмене задачи: {e}")

@click.command()
@click.option('--host', default='localhost', help='Host для сервера')
@click.option('--port', default=10007, help='Порт для сервера')
def main(host: str, port: int):
    """Запускает A2A сервер эксперта компании"""
    
    agent_executor = CompanyExpertExecutor()
    
    agent_card = AgentCard(
        name='Company Expert Agent',
        description='Эксперт по продуктам и услугам AI Solutions Corp. Может отвечать на вопросы о компании, выполнять расчеты и искать в документах.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        authentication={"schemes": ["basic"]},
        skills=[
            AgentSkill(
                id='company_expertise',
                name='Company Expertise',
                description='Экспертные знания о продуктах и услугах компании AI Solutions Corp',
                tags=['company', 'products', 'expertise'],
            ),
            AgentSkill(
                id='document_search',
                name='Document Search',
                description='Поиск информации в документах компании',
                tags=['search', 'documents'],
            ),
            AgentSkill(
                id='calculations',
                name='Calculations',
                description='Выполнение математических расчетов',
                tags=['math', 'calculations'],
            )
        ],
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    
    # Создаем A2A приложение
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    print(f"Запуск Company Expert Agent на {host}:{port}")
    print("Доступные навыки:")
    for skill in agent_card.skills:
        print(f"  - {skill.name}: {skill.description}")
    
    uvicorn.run(a2a_app.build(), host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()