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
	result := tensor.Equal(pred, tensor.Int(t.Data))
	acc := tensor.Mean(result)
	return variable.From(acc)
}
