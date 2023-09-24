package function

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

type Forwarder interface {
	Forward(x ...*variable.Variable) []*variable.Variable
	Backward(gy ...*variable.Variable) []*variable.Variable
}

type Function struct {
	In, Out []*variable.Variable
	Gen     int
	Forwarder
}

func (f *Function) Input() []*variable.Variable {
	return f.In
}

func (f *Function) Output() []*variable.Variable {
	return f.Out
}

func (f *Function) Generation() int {
	return f.Gen
}

// ApplyAndFirst applies the function and returns the first output variable.
func (f *Function) ApplyAndFirst(x ...*variable.Variable) *variable.Variable {
	return f.Apply(x...)[0]
}

// Apply applies the function
func (f *Function) Apply(x ...*variable.Variable) []*variable.Variable {
	y := f.Forward(x...)

	f.setCreator(y)
	f.In, f.Out, f.Gen = x, y, maxgen(x...)
	return f.Out
}

func (f Function) String() string {
	return fmt.Sprintf("%T%v", f.Forwarder, f.In)
}

func (f *Function) setCreator(y []*variable.Variable) {
	for i := range y {
		y[i].SetCreator(f)
	}
}

func maxgen(x ...*variable.Variable) int {
	var max int
	for _, v := range x {
		if max < v.Generation {
			max = v.Generation
		}
	}

	return max
}
