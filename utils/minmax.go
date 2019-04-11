package utils

import (
	"github.com/pkg/errors"
)

func ArrayMax(arr []float64) (float64, error) {
	if len(arr) == 0 {
		return 0, errors.New("Slice is empty")
	}
	max := arr[0]
	for _, v := range arr {
		if v > max {
			max = v
		}
	}
	return max, nil
}

func ArrayMin(arr []float64) (float64, error) {
	if len(arr) == 0 {
		return 0, errors.New("Slice is empty")
	}
	min := arr[0]
	for _, v := range arr {
		if v < min {
			min = v
		}
	}
	return min, nil
}
