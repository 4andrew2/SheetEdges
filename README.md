# SheetEdges

Скрипт распознает границы листа бумаги на фото.

Запуск - запустить main.py, указать путь к файлу в консоли. Результат работы сохраняется в result.jpg

Принцип работы - используется предобработка изображения при помощи Sobel Derivatives из cv2, свертки, затем применяется Canny Edge Detector и сглаживание найденного наибольшего контура.

Подробнее см в комментариях к коду.

Изображения приложены для демонстрации работы.
