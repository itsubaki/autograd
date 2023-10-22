package function

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *variable.Variable) *variable.Variable {
	argmax := matrix.From([][]int{matrix.Argmax(y.Data)})
	pred := matrix.Reshape(matrix.Shape(t.Data), argmax)
	result := matrix.F2(pred, t.Data, acc)
	return variable.New(matrix.Mean(result))
}

func acc(a, b float64) float64 {
	if math.Abs(a-b) < 1e-13 {
		return 1
	}

	return 0
}
