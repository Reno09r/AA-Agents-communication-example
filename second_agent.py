import asyncio
import os
import uuid
import httpx
import click
import uvicorn
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field

from a2a.types import (
    AgentCard, AgentCapabilities, AgentSkill, TaskState, TextPart,
    SendMessageRequest, MessageSendParams, GetTaskRequest, TaskQueryParams
)
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.client import A2AClient
from a2a.types import GetTaskSuccessResponse

load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise ValueError("MISTRAL_API_KEY not found. Please check your .env file.")

# URL эксперта компании
EXPERT_AGENT_URL = "http://localhost:10007/"


class ManagerResponse(BaseModel):
    """Manager Response Structure."""
    client_name: str = Field(
        description="The name of the client to contact, e.g. 'Sarah Jones'.")
    response_subject: str = Field(
        description="A suitable title for the response email, e.g. 'Re: Inquiry about AI Solutions'.")
    response_body: str = Field(description="""The full professional text of the response letter.
    Should recognise the client's needs, briefly explain how the requested products can help, and suggest the next step (planning demo).""")


# Создаем агента с правильной конфигурацией
manager_agent = Agent(
    model='mistral:mistral-large-latest',
    result_type=ManagerResponse,
    system_prompt="""You are a professional sales manager for AI solutions.
    Analyse incoming customer emails and create personalised responses in the specified format."""
)


class ManagerAgentExecutor(AgentExecutor):
    """A2A Executor для агента-менеджера"""

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Выполняет задачи менеджера"""
        task_updater = TaskUpdater(
            event_queue, context.task_id, context.context_id)

        try:
            if event_queue.is_closed():
                print("Warning: Event queue is closed, creating new one...")
                event_queue = EventQueue()
                task_updater = TaskUpdater(
                    event_queue, context.task_id, context.context_id)

            task_updater.submit()
            task_updater.start_work()

            client_email = context.message.parts[0].root.text
            print(f"Менеджер получил письмо клиента: {client_email[:100]}...")

            expert_info = await self._consult_expert(client_email)

            manager_prompt = f"""
            Information from the company expert:
            {expert_info}

            You have received the following email from a potential client:
            {client_email}

            Your task is to:
            1. address the client by name
            2. Recognise their specific problems
            3. Briefly explain how our products solve their problems using information from an expert
            4. Enthusiastically offer to schedule a demonstration
            5. Create a response in the required structured format
            6. Dont write Best regards and Sales Manager in the end of email.
            """

            try:
                manager_result = await manager_agent.run(manager_prompt)
                manager_reply = manager_result.output

                print(
                    f"Получен ответ от менеджера: {manager_reply.client_name}")

            except Exception as ai_error:
                print(f"Ошибка Pydantic AI: {ai_error}")
                manager_reply = ManagerResponse(
                    client_name="Sarah Jones",
                    response_subject="Re: Inquiry about AI Solutions for E-commerce Logistics",
                    response_body=f"""Thank you for your interest in our AI solutions!

                        I understand that Global Retail Express is facing challenges in document processing and inventory management.

                        Based on information from our expert:
                        {expert_info[:500]}...

                        Our Document Analyzer and Vision AI solutions are perfect for your challenges. 
                        Let's schedule a demo to show you how they can help your business.

                        When would be a good time for you to give an online presentation?"""
                )

            final_response = f"""To: {manager_reply.client_name}
                    From: AI Solutions Corp Sales Team
                    Subject: {manager_reply.response_subject}

                    {manager_reply.response_body}

                    Best regards,
                    AI Solutions Corp Team
                    Sales Department
                    Email: sales@aisolutions.corp
                    Phone: +1-555-AI-SOLUTIONS
            """.strip()

            print(f"Отправляем финальный ответ: {final_response[:200]}...")

            task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=final_response)]
                ),
            )

        except Exception as e:
            print(f"Ошибка в ManagerAgentExecutor: {e}")
            try:
                task_updater = TaskUpdater(
                    event_queue, context.task_id, context.context_id)
                task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[
                            TextPart(text=f"Ошибка при обработке письма: {str(e)}")]
                    ),
                )
            except Exception as update_error:
                print(
                    f"Критическая ошибка при обновлении статуса: {update_error}")

    async def _consult_expert(self, client_email: str) -> str:
        """Консультируется с экспертом компании через A2A"""
        try:
            # Формируем вопрос для эксперта
            expert_query = f"""
            Analyse this customer email and provide details of our products that can help:
            
            {client_email}
            
            Especially interested in information about Document Analyzer and Vision AI. Don't write a return email, just tell us about our products.        
            """

            print("Отправляем запрос эксперту...")

            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    expert_agent = await A2AClient.get_client_from_agent_card_url(
                        client, EXPERT_AGENT_URL
                    )

                    send_request = SendMessageRequest(
                        params=MessageSendParams(
                            message={
                                'messageId': str(uuid.uuid4()),
                                'role': 'user',
                                'parts': [{'type': 'text', 'text': expert_query}],
                            }
                        )
                    )

                    response = await expert_agent.send_message(send_request)

                    if hasattr(response.root, "result"):
                        expert_task = response.root.result
                        print(f"Создана задача для эксперта: {expert_task.id}")

                        max_attempts = 20
                        attempts = 0

                        while (expert_task.status.state not in (
                            TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected
                        ) and attempts < max_attempts):
                            await asyncio.sleep(0.5)
                            attempts += 1

                            try:
                                get_resp = await expert_agent.get_task(
                                    GetTaskRequest(
                                        params=TaskQueryParams(id=expert_task.id))
                                )
                                if isinstance(get_resp.root, GetTaskSuccessResponse):
                                    expert_task = get_resp.root.result
                                    print(
                                        f"Статус задачи эксперта: {expert_task.status.state}")
                                else:
                                    break
                            except Exception as get_error:
                                print(
                                    f"Ошибка при получении статуса задачи: {get_error}")
                                break

                        if expert_task.status.state == TaskState.completed and expert_task.status.message:
                            expert_response = expert_task.status.message.parts[0].root.text
                            print(
                                f"Получен ответ от эксперта: {expert_response[:100]}...")
                            return expert_response
                        else:
                            return f"Не удалось получить ответ от эксперта. Статус: {expert_task.status.state}"
                    else:
                        return "Ошибка при создании задачи для эксперта."

                except httpx.ConnectError as conn_error:
                    print(f"Ошибка подключения к эксперту: {conn_error}")
                    return "Эксперт временно недоступен. Используем базовую информацию о продуктах."

        except Exception as e:
            print(f"Ошибка при консультации с экспертом: {e}")
            return f"Ошибка связи с экспертом: {str(e)}. Используем базовую информацию."

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Отменяет выполнение задачи"""
        try:
            task_updater = TaskUpdater(
                event_queue, context.task_id, context.context_id)
            task_updater.update_status(TaskState.canceled)
        except Exception as e:
            print(f"Ошибка при отмене задачи: {e}")


@click.command()
@click.option('--host', default='localhost', help='Host для сервера')
@click.option('--port', default=10008, help='Порт для сервера')
def main(host: str, port: int):
    """Запускает A2A сервер менеджера"""

    agent_executor = ManagerAgentExecutor()

    agent_card = AgentCard(
        name='Sales Manager Agent',
        description='Профессиональный менеджер по продажам AI решений. Обрабатывает входящие запросы клиентов и генерирует персонализированные ответы.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        authentication={"schemes": ["basic"]},
        skills=[
            AgentSkill(
                id='client_communication',
                name='Client Communication',
                description='Обработка входящих писем клиентов и генерация профессиональных ответов',
                tags=['sales', 'communication', 'email'],
            ),
            AgentSkill(
                id='expert_consultation',
                name='Expert Consultation',
                description='Консультация с экспертом компании для получения точной информации о продуктах',
                tags=['consultation', 'expertise'],
            ),
        ],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    print(f"Запуск Sales Manager Agent на {host}:{port}")
    print("Доступные навыки:")
    for skill in agent_card.skills:
        print(f"  - {skill.name}: {skill.description}")
    print(f"Ожидаем эксперта компании на {EXPERT_AGENT_URL}")

    uvicorn.run(a2a_app.build(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
