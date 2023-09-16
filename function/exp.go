package function

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Exp(x *variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &ExpT{}}).Apply(x)
}

type ExpT struct {
	x variable.Data
}

func (f *ExpT) Forward(x variable.Data) variable.Data {
	f.x = x

	exp := func(a float64) float64 { return math.Exp(a) }
	y := vector.F(x, exp)
	return y
}

func (f *ExpT) Backward(gy variable.Data) variable.Data {
	dexp := func(a, b float64) float64 { return math.Exp(a) * b }
	grad := vector.F2(f.x, gy, dexp)
	return grad
}

func (f ExpT) String() string {
	return fmt.Sprintf("%T(%v)", f, f.x)
}
