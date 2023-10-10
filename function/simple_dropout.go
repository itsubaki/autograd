package function

import (
	"math/rand"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Dropout(ratio float64, s ...rand.Source) func(x ...*variable.Variable) *variable.Variable {
	return func(x ...*variable.Variable) *variable.Variable {
		if !variable.Config.Train {
			return x[0]
		}

		xs := variable.Shape(x[0])
		mask := matrix.Mask(matrix.Rand(xs[0], xs[1], s...), mask(ratio))
		return MulC(1.0/(1.0-ratio), Mul(x[0], variable.NewOf(mask...))) // y = x * mask / (1 - ratio)
	}
}

func mask(ratio float64) func(v float64) bool {
	return func(v float64) bool { return v > ratio }
}
