# test_a2a_system.py
import asyncio
import httpx
from uuid import uuid4
from pprint import pprint
from a2a.client import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams, GetTaskRequest, TaskQueryParams, TaskState
from a2a.types import GetTaskSuccessResponse

# Тестовое письмо клиента
CLIENT_EMAIL = """
Subject: Inquiry about AI Solutions for E-commerce Logistics

Hello,

My name is Sarah Jones, and I am the Operations Manager at 'Global Retail Express'.

We are rapidly expanding and facing challenges with processing a high volume of shipping documents and managing our new warehouse inventory.

I am particularly interested in your 'Document Analyzer' for automating invoice processing and your 'Vision AI' solution for inventory monitoring.

Could you please provide more details on these products and let me know if it's possible to schedule a demo?

Thank you,
Sarah Jones
Operations Manager
Global Retail Express
"""

async def wait_for_task_completion(client, task_id, max_wait_time=30):
    """Ждет завершения задачи с таймаутом"""
    attempts = 0
    max_attempts = max_wait_time * 2  # проверяем каждые 0.5 секунд
    
    while attempts < max_attempts:
        try:
            get_resp = await client.get_task(
                GetTaskRequest(params=TaskQueryParams(id=task_id))
            )
            
            if isinstance(get_resp.root, GetTaskSuccessResponse):
                task = get_resp.root.result
                
                if task.status.state in (TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected):
                    return task
                    
                print(f"Задача {task_id} в состоянии: {task.status.state}")
                
            await asyncio.sleep(0.5)
            attempts += 1
            
        except Exception as e:
            print(f"Ошибка при проверке статуса задачи: {e}")
            await asyncio.sleep(1)
            attempts += 2
    
    raise TimeoutError(f"Задача {task_id} не завершилась за {max_wait_time} секунд")

async def test_expert_agent():
    """Тестирует эксперта компании напрямую"""
    print("=== Тестирование Company Expert Agent ===")
    
    timeout_config = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        try:
            try:
                health_check = await httpx_client.get('http://localhost:10007/health', timeout=5.0)
                print(f"Expert health check: {health_check.status_code}")
            except Exception as health_error:
                print(f"Expert не отвечает на health check: {health_error}")
                return False
                
            expert_client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, 'http://localhost:10007'
            )
            
            question = "Tell us in detail about Document Analyzer and Vision AI products. What can they do and how much do they cost?"
            
            request = SendMessageRequest(
                params=MessageSendParams(
                    message={
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': question}],
                        'messageId': uuid4().hex,
                    }
                )
            )
            
            print("Отправляем вопрос эксперту...")
            response = await expert_client.send_message(request)
            
            if hasattr(response.root, "result"):
                task = response.root.result
                print(f"Создана задача: {task.id}")
                
                # Ждем завершения
                completed_task = await wait_for_task_completion(expert_client, task.id)
                
                if completed_task.status.state == TaskState.completed:
                    print("✅ Эксперт успешно ответил:")
                    if completed_task.status.message:
                        print(completed_task.status.message.parts[0].root.text)
                    return True
                else:
                    print(f"❌ Эксперт завершил работу с ошибкой: {completed_task.status.state}")
                    return False
            else:
                print("❌ Не удалось создать задачу для эксперта")
                return False
            
        except Exception as e:
            print(f"❌ Ошибка при тестировании эксперта: {e}")
            return False

async def test_manager_agent():
    """Тестирует менеджера с письмом клиента"""
    print("\n=== Тестирование Sales Manager Agent ===")
    
    timeout_config = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        try:
            try:
                health_check = await httpx_client.get('http://localhost:10008/health', timeout=5.0)
                print(f"Manager health check: {health_check.status_code}")
            except Exception as health_error:
                print(f"Manager не отвечает на health check: {health_error}")
                return False
                
            # Подключаемся к менеджеру
            manager_client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, 'http://localhost:10008'
            )
            
            request = SendMessageRequest(
                params=MessageSendParams(
                    message={
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': CLIENT_EMAIL}],
                        'messageId': uuid4().hex,
                    }
                )
            )
            
            print("Отправляем письмо клиента менеджеру...")
            response = await manager_client.send_message(request)
            
            if hasattr(response.root, "result"):
                task = response.root.result
                print(f"Создана задача: {task.id}")
                
                completed_task = await wait_for_task_completion(manager_client, task.id, max_wait_time=60)
                
                if completed_task.status.state == TaskState.completed:
                    print("✅ Менеджер успешно обработал письмо:")
                    if completed_task.status.message:
                        print(completed_task.status.message.parts[0].root.text)
                    return True
                else:
                    print(f"❌ Менеджер завершил работу с ошибкой: {completed_task.status.state}")
                    if completed_task.status.message:
                        print(f"Сообщение об ошибке: {completed_task.status.message.parts[0].root.text}")
                    return False
            else:
                print("❌ Не удалось создать задачу для менеджера")
                return False
            
        except Exception as e:
            print(f"❌ Ошибка при тестировании менеджера: {e}")
            return False

async def test_simple_calculation():
    """Тестирует простые расчеты через эксперта"""
    print("\n=== Тестирование расчетов ===")
    
    timeout_config = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        try:
            expert_client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, 'http://localhost:10007'
            )
            
            question = "What is 7 times 8? And also 15 plus 25?"
            
            request = SendMessageRequest(
                params=MessageSendParams(
                    message={
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': question}],
                        'messageId': uuid4().hex,
                    }
                )
            )
            
            print("Отправляем математический вопрос...")
            response = await expert_client.send_message(request)
            
            if hasattr(response.root, "result"):
                task = response.root.result
                completed_task = await wait_for_task_completion(expert_client, task.id)
                
                if completed_task.status.state == TaskState.completed:
                    print("✅ Расчеты выполнены:")
                    if completed_task.status.message:
                        print(completed_task.status.message.parts[0].root.text)
                    return True
                else:
                    print(f"❌ Ошибка при выполнении расчетов: {completed_task.status.state}")
                    return False
            else:
                print("❌ Не удалось создать задачу для расчетов")
                return False
            
        except Exception as e:
            print(f"❌ Ошибка при тестировании расчетов: {e}")
            return False

async def test_agent_availability():
    """Проверяет доступность обоих агентов"""
    print("=== Проверка доступности агентов ===")
    
    expert_available = False
    manager_available = False
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Проверяем эксперта
        try:
            expert_resp = await client.get('http://localhost:10007/')
            expert_available = expert_resp.status_code in [200, 404, 405]
            print(f"✅ Expert Agent доступен (статус: {expert_resp.status_code})")
        except Exception as e:
            print(f"❌ Expert Agent недоступен: {e}")
        
        # Проверяем менеджера
        try:
            manager_resp = await client.get('http://localhost:10008/')
            manager_available = manager_resp.status_code in [200, 404, 405]
            print(f"✅ Manager Agent доступен (статус: {manager_resp.status_code})")
        except Exception as e:
            print(f"❌ Manager Agent недоступен: {e}")
    
    return expert_available, manager_available

async def test_full_system():
    """Тестирует полную систему"""
    print("=== Тестирование полной A2A системы ===")
    
    # Проверяем доступность агентов
    expert_available, manager_available = await test_agent_availability()
    
    if not expert_available:
        print("❌ Expert Agent недоступен. Убедитесь, что он запущен на порту 10007")
        return
    
    if not manager_available:
        print("❌ Manager Agent недоступен. Убедитесь, что он запущен на порту 10008")
        return
    
    print("\n" + "="*50)
    expert_success = await test_expert_agent()
    
    print("\n" + "="*50)
    calc_success = await test_simple_calculation()
    
    print("\n" + "="*50)
    manager_success = await test_manager_agent()
    
    print("\n" + "="*50)
    print("=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
    print(f"Expert Agent: {'✅ РАБОТАЕТ' if expert_success else '❌ ОШИБКА'}")
    print(f"Calculations: {'✅ РАБОТАЕТ' if calc_success else '❌ ОШИБКА'}")
    print(f"Manager Agent: {'✅ РАБОТАЕТ' if manager_success else '❌ ОШИБКА'}")
    
    if expert_success and manager_success:
        print("\n🎉 Система A2A работает корректно!")
    else:
        print("\n⚠️  Обнаружены проблемы в системе")

async def run_interactive_test():
    """Интерактивное тестирование"""
    print("=== Интерактивное тестирование ===")
    
    while True:
        print("\nВыберите тест:")
        print("1. Проверить доступность агентов")
        print("2. Протестировать эксперта")
        print("3. Протестировать расчеты")
        print("4. Протестировать менеджера")
        print("5. Полное тестирование системы")
        print("6. Выход")
        
        choice = input("\nВведите номер теста: ").strip()
        
        if choice == '1':
            await test_agent_availability()
        elif choice == '2':
            await test_expert_agent()
        elif choice == '3':
            await test_simple_calculation()
        elif choice == '4':
            await test_manager_agent()
        elif choice == '5':
            await test_full_system()
        elif choice == '6':
            print("Завершение тестирования...")
            break
        else:
            print("Неверный выбор. Попробуйте еще раз.")

if __name__ == "__main__":
    print("🚀 Тестирование A2A системы")
    print("Убедитесь, что запущены оба агента:")
    print("1. Company Expert Agent на порту 10007")
    print("2. Sales Manager Agent на порту 10008")
    print("-" * 50)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        asyncio.run(run_interactive_test())
    else:
        asyncio.run(test_full_system())