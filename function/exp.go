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

	exp := func(a float64) float64 { return math.Exp(a) }
	y := vector.F(f.x, exp)
	return []variable.Data{y}
}

func (f *ExpT) Backward(gy ...variable.Data) []variable.Data {
	dexp := func(a, b float64) float64 { return math.Exp(a) * b }
	grad := vector.F2(f.x, gy[0], dexp)
	return []variable.Data{grad}
}
