# IntellegentPlacer
## Постановка задачи
Необходимо по фотографии одного или нескольких предметов на белом чистом листе A4 и многоугольника, нарисванного на другом белом чистом листе A4, определить, могут ли каким-либо способом данные предметы одновременно, но не перекрывая друг друга, поместиться в этот многоугольник. Все предметы, которые могут оказаться на фотографии, заранее известны.

### Вход\Выход
*Вход:* 
изображение в формате .jpg без сжатия, на котором в верхней части изображен многоугольник, а в нижней проверяемые предметы

*Выход:* 
**True**, в случае успеха и **False**, в случае, если предметы не могут быть умещены в многоугольник, или же изображение, поступившее на вход, не удовлетворяет требованиям. 
Результат выводится в файл "result\_[имя входного изображения\].txt"

## Требования
### Общие требования к изображениям
+ Все изображения в .jpg формате
+ Фотографии сделаны сверху, перпендикулярно плоскости объекта
+ Все фотографии сделаны на одно устройство и при одинаковом освещении, камера не менее 12 мегапикселей
+ Изображения сделаны в дневном освещении, объекты на фотографии имеют чёткую границу, отсутствуют засвеченные области и тени высокой интенсивности
+ Изображения цветные, без цветовой коррекции и наложения фильтров


### Требования к предметам:
+ Предметы полностью помещаются на лист формата A4
+ Предметы ярких цветов, не сливаются с листом бумаги
+ Предметы не перекрывают друг друга
+ Предметы не имеют общих границ (расстояние между предметами не менее 0.5 см)
+ Предмет может присутствовать на изображении только в единственном экземпляре


### Требования к поверхности:
+ Горизонтальная гладкая поверхность тёмного цвета
+ Поверхность одна для всех фотографий


### Требования к исходным данным
+ Исходные данные содержат изображения десяти различных предметов
+ Фотографии исходных предметов сделаны на чистом белом листе A4, при этом предметы находятся в центре листа 
+ Края листов бумаги на фото хорошо видны и не сливаются с фоном

### Требования к входным данным
+ На изображении находятся два чистых белых листа A4 в горизонтальной орентации, где один лист расположен строго под другим
+ Предметы распологаются на нижнем листе бумаги
+ Многоугольник нарисован на верхнем листе бумаги ярким цветом и хорошо контрастирует с белым фоном
+ Многоугольник выпуклый и имеет не более 6 вершин
+ Края листов бумаги на фото хорошо видны и не сливаются с фоном
+ Листы с предметами и нарисованым многоугольником не пересекаются и не имеют общих границ (расстояние не менее 1 см между границ)

## Данные
Изображения исходных объектов, доступны по [ссылке](https://drive.google.com/drive/folders/1a4XzSREjyc9MpqBEcfjzBQXmNRO5VQp6?usp=sharing)

Примеры входных данных  расположены в папках "true" и "false", в соответствии с ожидаемым результатом работы алгоритма, доступны по следующей [ссылке](https://drive.google.com/drive/folders/1Xi4-pHhXemMedG6puiFzId9kjIyVojrD?usp=sharing)
