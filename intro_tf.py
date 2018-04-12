import tensorflow as tf
# В качестве вводных примеров будем смотреть а статью https://tproger.ru/translations/text-classification-tensorflow-neural-networks/

"""
Каждое вычисление в TensorFlow представляется как граф потока данных. У него есть два элемента:

    Набор tf.Operation, который представляет единицы вычислений.
    Набор tf.Tensor, который представляет единицы данных.

"""

# Объявляем Граф вычислений
my_graph = tf.Graph()

# Чтобы инициализировать работу модели, мы должны содать сессию.
# Объект tf.Session инкапсулирует среду выполнения объектов Operation
# и оценки объектов Tensor

with tf.Session(graph=my_graph) as sess:

    # Создаем тензоры-константы
    x = tf.constant([1,3,6])
    y = tf.constant([1,1,1])

    # Операция сложения
    op = tf.add(x,y)

    """
        Для выполнения операций используется метод tf.Session.run(). 
        Он совершает один «шаг» вычислений TensorFlow, запуская 
        необходимый фрагмент графа для выполнения каждого объекта 
        Operation и оценки каждого Tensor, переданного в аргументе fetches
    """

    result = sess.run(fetches=op)
    print(result)