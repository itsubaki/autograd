package function

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func DropoutSimple(ratio float64, s ...randv2.Source) func(x ...*variable.Variable) *variable.Variable {
	return func(x ...*variable.Variable) *variable.Variable {
		if !variable.Config.Train {
			return x[0]
		}

		mask := tensor.Mask(tensor.Rand(x[0].Shape(), s...), mask(ratio))
		return MulC(1.0/(1.0-ratio), Mul(x[0], variable.NewFrom(mask))) // y = x * mask / (1 - ratio)
	}
}

func mask(ratio float64) func(v float64) bool {
	return func(v float64) bool { return v > ratio }
}
