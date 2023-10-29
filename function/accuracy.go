package function

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *variable.Variable) *variable.Variable {
	argmax := matrix.From([][]int{matrix.Argmax(y.Data)})
	pred := matrix.Reshape(matrix.Shape(t.Data), argmax)
	result := matrix.F2(pred, t.Data, variable.IsClose)
	return variable.New(matrix.Mean(result))
}
