package function

import (
	"math"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Exp(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &ExpT{}}).Apply(x...)[0]
}

type ExpT struct {
	x variable.Data
}

func (f *ExpT) Forward(x ...variable.Data) []variable.Data {
	f.x = x[0]

	y := vector.F(f.x, exp)
	return []variable.Data{y}
}

func (f *ExpT) Backward(gy ...variable.Data) []variable.Data {
	grad := vector.F2(f.x, gy[0], dexp)
	return []variable.Data{grad}
}

func exp(a float64) float64 { return math.Exp(a) }

func dexp(a, b float64) float64 { return math.Exp(a) * b }
