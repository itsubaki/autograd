package dot_test

import (
	"fmt"
	"regexp"

	"github.com/itsubaki/autograd/dot"
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleVar() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	x := variable.New(1)
	x.Name = "x"

	dotx := dot.Var(x)
	fmt.Println(re.ReplaceAllString(dotx, "**********"))
	fmt.Println(dotx == dot.Var(x))

	y := variable.New(1)
	y.Name = "y"

	doty := dot.Var(y)
	fmt.Println(dotx == doty)

	// Output:
	// "0x**********" [label="x", color=orange, style=filled]
	// true
	// false
}

func ExampleFunc() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	f0 := &F.Function{Forwarder: &F.SinT{}}
	for _, txt := range dot.Func(f0) {
		fmt.Println(re.ReplaceAllString(txt, "**********"))
	}

	f1 := &F.Function{Forwarder: &F.SinT{}}
	fmt.Println(dot.Func(f0)[0] == dot.Func(f1)[0])

	// Output:
	// "0x**********" [label="SinT", color=lightblue, style=filled, shape=box]
	// false
}

func Example_func() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	f := &F.Function{
		In:        []*variable.Variable{variable.New(1)},
		Out:       []*variable.Variable{variable.New(1)},
		Forwarder: &F.SinT{},
	}

	dotf := dot.Func(f)
	for _, txt := range dotf {
		fmt.Println(re.ReplaceAllString(txt, "**********"))
	}

	// Output:
	// "0x**********" [label="SinT", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
}

func ExampleGraph() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	x := variable.New(1.0)
	x.Name = "x"

	y := F.Sin(x)
	y.Name = "y"

	for _, txt := range dot.Graph(y) {
		fmt.Println(re.ReplaceAllString(txt, "**********"))
	}

	// Output:
	// digraph g {
	// "0x**********" [label="y", color=orange, style=filled]
	// "0x**********" [label="SinT", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="x", color=orange, style=filled]
	// }
}
