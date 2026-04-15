package model

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Activation = F.ReLU
	_ Activation = F.Sigmoid
	_ Activation = F.Softmax(1)
	_ Activation = F.Tanh
)

// Activation represents an activation function used in a model.
type Activation func(x ...*variable.Variable) *variable.Variable
