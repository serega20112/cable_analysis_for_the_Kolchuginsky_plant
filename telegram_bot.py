import telebot
from cable_analyzer import CableAnalyzer
import cv2
# Ваши токены и настройки
BOT_TOKEN = "8175430341:AAH5giS2p020esrFuUt0GzeBb0HQPr9x5lw"
PIXELS_PER_MM = None  # Задайте значение, если известно, например, 5

# Создание экземпляра анализатора
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Отправьте мне изображение кабеля для анализа.")

def analyze_image(update, context):
    image_bytes = context.bot.get_file(update.message.photo[-1].file_id).download_as_bytearray()
    analyzer = CableAnalyzer()
    result = analyzer.analyze_image_bytes(image_bytes)
    if result:
        num_cores = result["num_cores"]
        diameter_px = result["diameter_px"]
        diameter_mm = result["diameter_mm"]
        processed_image = result["processed_image"]

        # Сохраняем обработанное изображение в файл
        analyzer.save_processed_image(processed_image, "processed_image.jpg")

        # Отправляем результат пользователю
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Количество сердечников: {num_cores}\nДиаметр (px): {diameter_px}\nДиаметр (мм): {diameter_mm}")
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=open("processed_image.jpg", "rb"))

def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, analyze_image))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()