# src/main.py
import json
from dotenv import load_dotenv

from src.services.external_api import mock_external_api_call, format_docs_for_prompt
from src.chains.processing_chain import create_processing_chain
from src.llm.client import get_llm_client # <<< ИЗМЕНЕНИЕ: импортируем нашу фабрику

def run_service():
    """
    Основная функция для запуска сервиса.
    """
    load_dotenv()
    
    # 1. Инициализация модели через нашу фабрику
    try:
        llm = get_llm_client()
    except ValueError as e:
        print(f"Ошибка инициализации LLM: {e}")
        return
    
    # 2. Получение запроса от пользователя
    user_query = "срок подачи заявления в ифнс по смене директора"
    print(f"\nПользовательский запрос: '{user_query}'\n")

    # 3. Отправка запроса по API (post) и получение результата выдачи
    documents = mock_external_api_call(user_query)
    formatted_context = format_docs_for_prompt(documents)
    
    # 4. Создание и запуск цепочки обработки
    # Этот код не меняется, так как create_processing_chain просто принимает готовый llm объект
    print("--- Запуск цепочки обработки LangChain ---")
    processing_chain = create_processing_chain(llm, documents)
    
    # Запускаем цепочку
    final_result = processing_chain.invoke({
        "question": user_query,
        "context": formatted_context
    })
    
    # 5. Вывод результата
    print("\n--- Финальный результат ---")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_service()