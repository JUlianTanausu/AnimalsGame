package com.example.juliantanausuvillar.animalsgame

import java.util.Arrays

fun indicesOfMinElements(orig: FloatArray, nummax: Int): IntArray {
    val copy = Arrays.copyOf(orig, orig.size)
    Arrays.sort(copy)
    val honey = Arrays.copyOfRange(copy, 0, nummax)
    val result = IntArray(nummax)
    var resultPos = 0
    for (i in orig.indices) {
        val onTrial = orig[i]
        val index = Arrays.binarySearch(honey, onTrial)
        if (index < 0) continue
        result[resultPos++] = i
    }
    return result
}

fun mostFrequentElement(a: IntArray): Int {
    var count = 1
    var tempCount: Int
    var popular = a[0]
    for (i in 0 until a.size - 1) {
        val temp = a[i]
        tempCount = 0
        for (j in 1 until a.size) {
            if (temp == a[j])
                tempCount++
        }
        if (tempCount > count) {
            popular = temp
            count = tempCount
        }
    }
    return popular
}