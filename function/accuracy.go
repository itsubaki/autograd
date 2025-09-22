package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// Accuracy returns the accuracy of the prediction.
// The return values cannot be backpropagated.
func Accuracy(y, t *variable.Variable) *variable.Variable {
	expected := tensor.Int(t.Data)
	actual := tensor.Reshape(tensor.Argmax(y.Data, 1), t.Shape()...)
	result := tensor.Equal(actual, expected)
	return variable.From(tensor.Mean(result))
}
