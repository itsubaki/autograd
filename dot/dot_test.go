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

	fmt.Println(re.ReplaceAllString(dot.Var(x), "**********"))
	fmt.Println(re.ReplaceAllString(dot.Var(x, dot.Opt{Verbose: true}), "**********"))
	fmt.Println(dot.Var(x) == dot.Var(x))

	y := variable.New(1)
	y.Name = "x"
	fmt.Println(dot.Var(x) == dot.Var(y))

	// Output:
	// "0x**********" [label="x", color=orange, style=filled]
	// "0x**********" [label="x[1]", color=orange, style=filled]
	// true
	// false
}

func ExampleFunc() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	f0 := &variable.Function{Forwarder: &variable.SinT{}}
	for _, txt := range dot.Func(f0) {
		fmt.Println(re.ReplaceAllString(txt, "**********"))
	}

	f1 := &variable.Function{Forwarder: &variable.SinT{}}
	fmt.Println(dot.Func(f0)[0] == dot.Func(f1)[0])

	// Output:
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// false
}

func Example_func() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	f := &variable.Function{
		Input:     []*variable.Variable{variable.New(1)},
		Output:    []*variable.Variable{variable.New(1)},
		Forwarder: &variable.SinT{},
	}

	dotf := dot.Func(f)
	for _, txt := range dotf {
		fmt.Println(re.ReplaceAllString(txt, "**********"))
	}

	// Output:
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
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
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="x", color=orange, style=filled]
	// }
}

func ExampleGraph_composite() {
	var re = regexp.MustCompile("[0-9a-f]{10}")

	x := variable.New(1.0)
	y := F.Sin(x)
	z := F.Cos(y)
	x.Name = "x"
	y.Name = "y"
	z.Name = "z"

	for _, txt := range dot.Graph(z) {
		fmt.Println(re.ReplaceAllString(txt, "**********"))
	}

	// Output:
	// digraph g {
	// "0x**********" [label="z", color=orange, style=filled]
	// "0x**********" [label="Cos", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="y", color=orange, style=filled]
	// "0x**********" [label="Sin", color=lightblue, style=filled, shape=box]
	// "0x**********" -> "0x**********"
	// "0x**********" -> "0x**********"
	// "0x**********" [label="x", color=orange, style=filled]
	// }
}
