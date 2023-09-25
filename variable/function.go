package variable

import "fmt"

type Forwarder interface {
	Forward(x ...*Variable) []*Variable
	Backward(gy ...*Variable) []*Variable
}

type Function struct {
	In, Out []*Variable
	Gen     int
	Forwarder
}

func (f *Function) Input() []*Variable {
	return f.In
}

func (f *Function) Output() []*Variable {
	return f.Out
}

func (f *Function) Generation() int {
	return f.Gen
}

// ApplyAndFirst applies the function and returns the first output
func (f *Function) ApplyAndFirst(x ...*Variable) *Variable {
	return f.Apply(x...)[0]
}

// Apply applies the function
func (f *Function) Apply(x ...*Variable) []*Variable {
	y := f.Forward(x...)

	f.setCreator(y)
	f.In, f.Out, f.Gen = x, y, maxgen(x...)
	return f.Out
}

func (f Function) String() string {
	return fmt.Sprintf("%T%v", f.Forwarder, f.In)
}

func (f *Function) setCreator(y []*Variable) {
	for i := range y {
		y[i].SetCreator(f)
	}
}

func maxgen(x ...*Variable) int {
	var max int
	for _, v := range x {
		if max < v.Generation {
			max = v.Generation
		}
	}

	return max
}
