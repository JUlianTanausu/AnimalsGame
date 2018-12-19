package com.example.juliantanausuvillar.animalsgame

class KNearestNeighbors(private val k: Int) {

    private val X = mutableListOf<FloatArray>()
    private val y = mutableListOf<Int>()

    fun fit(x_train: FloatArray, y_train: Int) {
        X.add(x_train)
        y.add(y_train)
    }

    private fun euclideanDistance(x_1: FloatArray, x_2: FloatArray): Float {
        var sum = 0.0

        for (i in 0 until x_1.size) {
            sum += Math.pow((x_1[i] - x_2[i]).toDouble(), 2.0)
        }

        return Math.sqrt(sum).toFloat()
    }

    fun predict(x_test: FloatArray): Int {
        var distances = FloatArray(X.size)

        for (i in 0 until X.size) {
            distances[i] = euclideanDistance(x_test, X[i])
        }

        val topIndices = indicesOfMinElements(distances, k)

        val classes = IntArray(k)
        topIndices.forEachIndexed { i, topIndex ->
            classes[i] = y[topIndex]
        }

        return mostFrequentElement(classes)
    }

}