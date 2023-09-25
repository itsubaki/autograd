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

type Opt struct {
	Verbose bool
}

func Var(v *variable.Variable, opt ...Opt) string {
	if len(opt) > 0 && opt[0].Verbose {
		return fmt.Sprintf(varfmt, v, v)
	}

	return fmt.Sprintf(varfmt, v, v.Name)
}

func Func(f *variable.Function) []string {
	str := fmt.Sprintf("%s", f)
	begin, end := strings.Index(str, "."), strings.Index(str, "T[")
	out := []string{fmt.Sprintf(fncfmt, f, str[begin+1:end])}

	for _, v := range f.Input() {
		out = append(out, fmt.Sprintf(arrow, v, f))
	}

	for _, v := range f.Output() {
		out = append(out, fmt.Sprintf(arrow, f, v))
	}

	return out
}

func Graph(v *variable.Variable, opt ...Opt) []string {
	seen := make(map[*variable.Function]bool)
	fs := addFunc(make([]*variable.Function, 0), v.Creator, seen)

	out := append([]string{"digraph g {"}, Var(v, opt...))
	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]
		out = append(out, Func(f)...)

		x := f.Input()
		for i := range x {
			out = append(out, Var(x[i], opt...))

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
