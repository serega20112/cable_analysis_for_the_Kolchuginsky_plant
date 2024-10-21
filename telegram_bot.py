import telebot
from cable_analyzer import CableAnalyzer

# Ваши токены и настройки
BOT_TOKEN = "8175430341:AAH5giS2p020esrFuUt0GzeBb0HQPr9x5lw"
PIXELS_PER_MM = None  #  Задайте значение, если известно, например, 5

# Создание экземпляра анализатора
analyzer = CableAnalyzer(pixels_per_mm=PIXELS_PER_MM)

# Создание объекта бота
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        # 1. Получение информации о файле изображения
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)

        # 2. Загрузка файла изображения в виде байтов
        file = bot.download_file(file_info.file_path)

        # 3. Анализ изображения
        analysis_result = analyzer.analyze_image_bytes(file)

        # 4. Формирование ответа
        response = create_response(analysis_result)

        # 5. Отправка ответа пользователю
        bot.send_message(message.chat.id, response)

    except Exception as e:
        print(f"Ошибка: {e}")
        bot.send_message(message.chat.id, "Произошла ошибка при обработке изображения.")



def create_response(analysis_result):
    """Формирует текстовое описание анализа.

    Args:
        analysis_result (dict): Результат анализа.

    Returns:
        str: Текстовое описание.
    """
    response = "Результат анализа:\n"
    if analysis_result.get("num_cores"):
        response += f"• Количество жил: {analysis_result['num_cores']}\n"
    if analysis_result.get("diameter_px"):
        response += f"• Диаметр кабеля (пиксели): {analysis_result['diameter_px']:.1f}\n"
    if analysis_result.get("diameter_mm"):
        response += f"• Диаметр кабеля (мм): {analysis_result['diameter_mm']:.1f}\n"
    # ... добавьте другие характеристики ...

    if response == "Результат анализа:\n":
        response = "Не удалось проанализировать изображение кабеля."

    return response


bot.polling()
