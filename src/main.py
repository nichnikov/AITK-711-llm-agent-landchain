# Main script for running the LangChain service and demonstrating its functionality
# src/main.py
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.services.external_api import mock_external_api_call, format_docs_for_prompt
from src.chains.processing_chain import create_processing_chain

def run_service():
    """
    Основная функция для запуска сервиса.
    """
    load_dotenv()
    
    # 1. Инициализация модели
    # Убедитесь, что OPENAI_API_KEY есть в .env файле
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: Переменная окружения OPENAI_API_KEY не найдена.")
        print("Пожалуйста, создайте файл .env и добавьте в него OPENAI_API_KEY='ваш_ключ'")
        return

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 2. Получение запроса от пользователя
    user_query = "срок подачи заявления в ифнс по смене директора"
    print(f"Пользовательский запрос: '{user_query}'\n")

    # 3. Отправка запроса по API (post) и получение результата выдачи
    documents = mock_external_api_call(user_query)
    formatted_context = format_docs_for_prompt(documents)
    
    # 4. Создание и запуск цепочки обработки
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