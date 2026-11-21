package variable_test

import (
	"fmt"
	"testing"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleNograd() {
	f := func() {
		x := variable.New(3)
		y := variable.Square(x)
		y.Backward()

		fmt.Println("gx: ", x.Grad)
		fmt.Println()
	}

	fmt.Println("backprop:", variable.Config.EnableBackprop)
	f()

	func() {
		defer variable.Nograd().End()

		fmt.Println("backprop:", variable.Config.EnableBackprop)
		f()
	}()

	fmt.Println("backprop:", variable.Config.EnableBackprop)
	f()

	// Output:
	// backprop: true
	// gx:  variable(6)
	//
	// backprop: false
	// gx:  <nil>
	//
	// backprop: true
	// gx:  variable(6)
}

func ExampleTestMode() {
	fmt.Println("train:", variable.Config.Train)

	func() {
		defer variable.TestMode().End()

		fmt.Println("train:", variable.Config.Train)
	}()

	fmt.Println("train:", variable.Config.Train)

	// Output:
	// train: true
	// train: false
	// train: true
}

func ExampleNograd_nested() {
	func() {
		defer variable.Nograd().End()

		func() {
			defer variable.Nograd().End()

			fmt.Println(variable.Config.EnableBackprop)
		}()

		fmt.Println(variable.Config.EnableBackprop)
	}()

	fmt.Println(variable.Config.EnableBackprop)

	// Output:
	// false
	// false
	// true
}

func TestNograd_nested(t *testing.T) {
	cases := []struct {
		name    string
		initial bool
		depth   int
	}{
		{
			name:    "enabled single scope",
			initial: true,
			depth:   1,
		},
		{
			name:    "enabled double scope",
			initial: true,
			depth:   2,
		},
		{
			name:    "enabled triple scope",
			initial: true,
			depth:   3,
		},
		{
			name:    "disabled single scope",
			initial: false,
			depth:   1,
		},
		{
			name:    "disabled triple scope",
			initial: false,
			depth:   3,
		},
	}

	previous := variable.Config
	defer func() { variable.Config = previous }()

	for _, c := range cases {
		variable.Config.EnableBackprop = c.initial
		spans := make([]*variable.Span, 0, c.depth)

		for level := range c.depth {
			spans = append(spans, variable.Nograd())

			if variable.Config.EnableBackprop {
				t.Fatalf("%s: level %d did not disable EnableBackprop", c.name, level)
			}
		}

		for level := c.depth - 1; level >= 0; level-- {
			spans[level].End()

			want := level == 0 && c.initial
			if got := variable.Config.EnableBackprop; got != want {
				t.Fatalf("%s: after ending level %d, got %v, want %v", c.name, level, got, want)
			}

			if y := variable.Square(variable.New(3)); (y.Creator != nil) != want {
				t.Errorf("%s: after ending level %d, graph tracking does not match %v", c.name, level, want)
			}
		}
	}
}

func ExampleTestMode_nested() {
	func() {
		defer variable.TestMode().End()

		func() {
			defer variable.TestMode().End()

			fmt.Println(variable.Config.Train)
		}()

		fmt.Println(variable.Config.Train)
	}()

	fmt.Println(variable.Config.Train)

	// Output:
	// false
	// false
	// true
}

func TestTestMode_nested(t *testing.T) {
	cases := []struct {
		name    string
		initial bool
		depth   int
	}{
		{
			name:    "enabled single scope",
			initial: true,
			depth:   1,
		},
		{
			name:    "enabled double scope",
			initial: true,
			depth:   2,
		},
		{
			name:    "enabled triple scope",
			initial: true,
			depth:   3,
		},
		{
			name:    "disabled single scope",
			initial: false,
			depth:   1,
		},
		{
			name:    "disabled triple scope",
			initial: false,
			depth:   3,
		},
	}

	previous := variable.Config
	defer func() { variable.Config = previous }()

	for _, c := range cases {
		variable.Config.Train = c.initial
		spans := make([]*variable.Span, 0, c.depth)
		x := variable.Ones(2)
		dropout := F.DropoutSimple(0.5)

		for level := range c.depth {
			spans = append(spans, variable.TestMode())

			if variable.Config.Train {
				t.Fatalf("%s: level %d did not disable Train", c.name, level)
			}
		}

		for level := c.depth - 1; level >= 0; level-- {
			spans[level].End()

			want := level == 0 && c.initial
			if got := variable.Config.Train; got != want {
				t.Fatalf("%s: after ending level %d, got %v, want %v", c.name, level, got, want)
			}

			if got := dropout(x); !tensor.IsCloseAll(got.Data, x.Data) != want {
				t.Errorf("%s: after ending level %d, dropout training mode does not match %v", c.name, level, want)
			}
		}
	}
}

func TestNograd_backward(t *testing.T) {
	cases := []struct {
		name    string
		initial bool
	}{
		{
			name:    "enabled before scope",
			initial: true,
		},
		{
			name:    "disabled before scope",
			initial: false,
		},
	}

	previous := variable.Config
	defer func() { variable.Config = previous }()

	for _, c := range cases {
		variable.Config.EnableBackprop = true
		x := variable.New(3)
		y := variable.Square(x)
		variable.Config.EnableBackprop = c.initial

		span := variable.Nograd()
		y.Backward()

		if x.Grad == nil || !tensor.IsCloseAll(x.Grad.Data, variable.New(6).Data) {
			t.Errorf("%s: gradient = %v, want 6", c.name, x.Grad)
		}

		if variable.Config.EnableBackprop {
			t.Fatalf("%s: backward re-enabled tracking inside Nograd", c.name)
		}

		span.End()

		if got := variable.Config.EnableBackprop; got != c.initial {
			t.Errorf("%s: restored tracking = %v, want %v", c.name, got, c.initial)
		}
	}
}
