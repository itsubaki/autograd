package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *variable.Variable) *variable.Variable {
	argmax := tensor.Argmax(y.Data, 1)
	pred := tensor.Reshape(argmax, t.Shape()...)
	result := tensor.F2(tensor.Float64(pred), t.Data, variable.IsClose)
	return variable.From(tensor.Mean(result))
}
