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
	result := matrix.F2(pred, t.Data, isclose)
	return variable.New(matrix.Mean(result))

}

func isclose(a, b float64) float64 {
	atol, rtol := 1e-08, 1e-05
	if math.Abs(a-b) < atol+rtol*math.Abs(b) {
		return 1
	}

	return 0
}
