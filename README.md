# Intelligent Placer
## Постановка задачи
Необходимо по фотографии одного или нескольких предметов на светлой поверхности и многоугольника, который расположен на белом листе A4, определить, могут ли каким-либо способом данные предметы одновременно, но не перекрывая друг друга, поместиться в этот многоугольник. Все предметы, которые могут оказаться на фотографии, заранее известны.

### Вход\Выход
*Вход:* 
изображение в формате .jpg без сжатия, на котором в верхней части изображен многоугольник, а в нижней проверяемые предметы

*Выход:* 
**True**, в случае успеха и **False**, в случае, если предметы не могут быть умещены в многоугольник, или же изображение, поступившее на вход, не удовлетворяет требованиям. 
В случае положительного результата сохраняется найденный алгоритмом способ расположения предметов в многоугольнике по заданному пути в формате "True_{имя исходного файла}.jpg"

## Требования
### Общие требования к изображениям
+ Все изображения в .jpg формате
+ Фотографии сделаны сверху, перпендикулярно плоскости объекта. Отклонение прямой зрения объектива от перпендикуляра к горизонтальной плоскости не превышает 10 градусов.
+ Все фотографии сделаны на одно устройство и при одинаковом освещении, камера не менее 12 мегапикселей
+ Изображения сделаны в дневном освещении, объекты на фотографии имеют чёткую границу, отсутствуют засвеченные области и тени высокой интенсивности
+ Изображения цветные, без цветовой коррекции и наложения фильтров


### Требования к предметам:
+ Предметы не сливаются с поверхностью (параметры цветовой модели RGB не могут быть одновременно превышать значение 220. Например, rgb(240,240,240) - светло серый цвет, который будет сливаться с белом листом бумаги)
+ Предметы не перекрывают друг друга
+ Предметы не имеют общих границ (расстояние между предметами не менее 0.5 см (19 пикселей))
+ Предмет может присутствовать на изображении только в единственном экземпляре


### Требования к поверхности:
+ Гладкая, однотонная белая поверхность
+ Поверхность одна для всех фотографий


### Требования к исходным данным
+ Исходные данные содержат изображения десяти различных предметов
+ Фотографии исходных предметов сделаны на чистом белом листе A4, при этом предметы находятся в центре листа 
+ Края листов бумаги на фото хорошо видны 

### Требования к входным данным
+ Верхнюю половину изображения занимает белый лист бумаги A4 в горизонтальной ориентации, на котором нарисован многоугольник ярким цветом 
+ Края листа бумаги на фото хорошо видны 
+ Нижнюю половину изображения занимают предметы

## Данные
Изображения исходных объектов, доступны по [ссылке](https://drive.google.com/drive/folders/1a4XzSREjyc9MpqBEcfjzBQXmNRO5VQp6?usp=sharing)

Примеры входных данных  расположены в папках "true" и "false", в соответствии с ожидаемым результатом работы алгоритма, доступны по следующей [ссылке](https://drive.google.com/drive/folders/1Xi4-pHhXemMedG6puiFzId9kjIyVojrD?usp=sharing)

## План работы алгоритма
### Обработка изображений заранее известных предметов
1. Бинаризировать изображения с применением threshold_otsu и морфологических операций
2. Получить свойства найденной маски предмета
3. Применить маску к изображению и обрезать изображение по области маски
4. Нанести контуры найденной маски на исходное изображение (для более наглядной проверки точности определения маски)

### Поиск многоугольника и предметов на входном изображении
1. Сгладить изображения с использованием размытия по Гауссу
2. Найти границы объектов с использованием детектора Кэнни
3. Найти маску объектов с помощью морфологических операций
4. Разрезать изображение на две части относительно середины вертикали
5. На изображении полученном из верхней части находим многоугольник, а из нижней части предметы
6. Получаем свойства найденных предметов и многоугольника

### Решение задачи размещения предметов в многоугольнике
1. Найти сумму площадей найденных предметов и сравнить с площадью многоугольника
2. Если сумма площадей предметов превосходит площадь многоугольника, то прервать работу алгоритма и вернуть **False**
3. С помощью параллельного переноса сдвигаем по одному предмету внутри маски многоугольника на заданный шаг
4. Проверяем удачно ли размещен предмет внутри маски прямоугольника с использованием побитовой операции AND, если да, то предмет закрепляется а этом месте, если нет, то возвращаемся к шагу 3.
5. При удачном размещении всех предметов алгоритм возвращает True, а изображение найденного алгоритмом варианта размещения сохраняется по заданному пути

## Результаты работы алгоритма
Алгоритм был запущен на входном датасете, состоящем из 22 изображений и дал неверный ответ на 4 из низ. Можно определить, что в данном эксперименте **точность алгоритма равна 0.8181**. 

Эти ошибки были ожидаемыми, так как во всех ошибочных ситуациях удачного размещения можно добиться вращением предметов, что на данном этапе не учитывается алгоритмом

## Будущие планы
+ Учитывать исходные изображения в обработке входных данных
+ Добавить в алгоритм размещения возможность вращения предметов
+ При отрицательном ответе алгоритма повторить процесс размещения, но уже начиная с другого предмета. Таким образом перебрать все возможные варианты размещений
