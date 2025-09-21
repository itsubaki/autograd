package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *variable.Variable) *variable.Variable {
	argmax := matrix.New(f64(matrix.Argmax(y.Data)))
	pred := matrix.Reshape(argmax, t.Shape()...)
	result := matrix.F2(pred, t.Data, variable.IsClose)
	return variable.New(matrix.Mean(result))
}

func f64(x []int) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}

	return out
}
