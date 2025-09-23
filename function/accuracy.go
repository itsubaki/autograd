package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *variable.Variable) *variable.Variable {
	label := tensor.Int(t.Data)
	pred := tensor.Reshape(tensor.Argmax(y.Data, 1), t.Shape()...)
	hit := tensor.Equal(pred, label)
	return variable.From(tensor.Mean(hit))
}
