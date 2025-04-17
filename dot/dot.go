package dot

import (
	"fmt"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

const (
	varfmt = "\"%p\" [label=\"%v\", color=orange, style=filled]"
	fncfmt = "\"%p\" [label=\"%s\", color=lightblue, style=filled, shape=box]"
	arrow  = "\"%p\" -> \"%p\""
)

type Opts struct {
	Verbose bool
}

func Var(v *variable.Variable, opts ...Opts) string {
	if len(opts) > 0 && opts[0].Verbose {
		return fmt.Sprintf(varfmt, v, v)
	}

	return fmt.Sprintf(varfmt, v, v.Name)
}

func Func(f *variable.Function) []string {
	s := f.String()
	begin, end := strings.Index(s, "."), strings.Index(s, "T[")
	out := []string{fmt.Sprintf(fncfmt, f, s[begin+1:end])}

	for _, x := range f.Input {
		out = append(out, fmt.Sprintf(arrow, x, f))
	}

	for _, y := range f.Output {
		out = append(out, fmt.Sprintf(arrow, f, y))
	}

	return out
}

func Graph(v *variable.Variable, opts ...Opts) []string {
	seen := make(map[*variable.Function]bool)
	fs := addFunc(make([]*variable.Function, 0), v.Creator, seen)

	out := append([]string{"digraph g {"}, Var(v, opts...))
	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]
		out = append(out, Func(f)...)

		x := f.Input
		for i := range x {
			out = append(out, Var(x[i], opts...))

			if x[i].Creator != nil {
				fs = addFunc(fs, x[i].Creator, seen)
			}
		}
	}

	out = append(out, "}")
	return out
}

func addFunc(fs []*variable.Function, f *variable.Function, seen map[*variable.Function]bool) []*variable.Function {
	if _, ok := seen[f]; ok {
		return fs
	}

	seen[f] = true
	fs = append(fs, f)
	return fs
}
