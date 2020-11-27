package stats

import (
	"encoding/csv"
	"os"
	"testing"
)

func TestWriter_Init(t *testing.T) {
	type fields struct {
		base   string
		file   *os.File
		writer *csv.Writer
	}
	type args struct {
		folder string
		root   bool
		id     string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "Create folder correctly",
			fields: fields{
				base: "/Users/Raphael/go/src/github.com/ldsec/spindle/simul/nn/decentralization",
			},
			args: args{
				folder: "/test",
				root:   false,
				id:     "yoyoy",
			},
		},
		{
			name: "Create master correctly",
			fields: fields{
				base: "/Users/Raphael/go/src/github.com/ldsec/spindle/simul/nn/decentralization",
			},
			args: args{
				folder: "/test",
				root:   true,
				id:     "123",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := NewStatsWriter(tt.fields.base)
			w.Init(tt.args.root, tt.args.id)

			if tt.name == "Create folder correctly" {
				if _, err := os.Stat(tt.fields.base + tt.args.folder); os.IsNotExist(err) {
					t.Error("Folder not created")
				}
				if _, err := os.Stat(tt.fields.base + tt.args.folder); os.IsNotExist(err) {
					t.Error("Folder not created")
				}
			} else {
				if _, err := os.Stat(tt.fields.base + tt.args.folder); os.IsNotExist(err) {
					t.Error("Folder not created")
				}
				if _, err := os.Stat(tt.fields.base + tt.args.folder + "/master_123.csv"); os.IsNotExist(err) {
					t.Error("Folder not created")
				}
			}

		})
	}
}

func TestWriter_Write(t *testing.T) {
	type fields struct {
		base   string
		file   *os.File
		writer *csv.Writer
	}
	type args struct {
		value []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "Create master correctly",
			fields: fields{
				base: "/Users/Raphael/go/src/github.com/ldsec/spindle/simul/nn/decentralization",
			},
			args: args{
				value: []float64{1, 2, 3, 4, 5},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &Writer{
				base:   tt.fields.base,
				file:   tt.fields.file,
				writer: tt.fields.writer,
			}
			w.Init(true, "hello")
			w.Write(tt.args.value)
		})
	}
}
