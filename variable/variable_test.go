package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleVariable() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(v)

	// Output:
	// variable[1 2 3 4]
}

func ExampleVariable_Name() {
	v := variable.New(1, 2, 3, 4)
	v.Name = "v"
	fmt.Println(v)

	// Output:
	// v[1 2 3 4]
}

func ExampleConst() {
	fmt.Println(variable.Const(1))

	// Output:
	// variable[1]
}

func ExampleOneLike() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(variable.OneLike(v))

	// Output:
	// variable[1 1 1 1]
}

func ExampleVariable_Backward() {
	x := variable.New(1.0)
	x.Backward()
	fmt.Println(x.Grad)

	x.Cleargrad()
	x.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1]
	// variable[1]
}

type Creator struct {
	In, Out []*variable.Variable
	Gen     int
}

func (c *Creator) Input() []*variable.Variable {
	return c.In
}

func (c *Creator) Output() []*variable.Variable {
	return c.Out
}

func (c *Creator) Generation() int {
	return c.Gen
}

func (c *Creator) Backward(gy ...*variable.Variable) []*variable.Variable {
	return gy
}

func ExampleVariable_SetCreator() {
	v := variable.New(1, 2, 3, 4)
	w := variable.New(5, 6, 7, 8)

	c := &Creator{
		Gen: 100,
		In:  []*variable.Variable{v},
		Out: []*variable.Variable{w},
	}
	v.SetCreator(c)
	w.Backward() // w.Grad = variable[1 1 1 1]
	v.Backward() // v.Grad = variable[1 1 1 1] + w.Grad

	fmt.Println(v.Generation)
	fmt.Println(v.Grad)

	// Output:
	// 101
	// variable[2 2 2 2]
}
