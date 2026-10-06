// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package recipe

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"testing"
)

const (
	nfdPubNFD             = "nfd"
	nfdPubNetworkOperator = "network-operator"
	nfdPubMellanoxLabel   = "feature.node.kubernetes.io/pci-15b3.present"
	nfdPubMellanoxBare    = "pci-15b3.present"
)

// nfdPubChartRule is a NodeFeatureRule that a registry chart renders itself,
// outside recipes/components/*/manifests. nfd-master evaluates it against the
// features nfd-worker publishes, so the feature it reads must not appear in
// worker.config.core.noPublishFeatures.
type nfdPubChartRule struct {
	component    string
	enablePath   []string
	chartDefault bool
	feature      string
}

// nfdPubChartRules comes from the chart templates at the registry pins on
// 2026-10-06: network-operator 26.4.1 templates/nodefeaturerules.yaml,
// gpu-operator v26.7.1 templates/nodefeaturerules.yaml, k8s-nim-operator 3.1.0
// templates/node-feature-rule.yaml. Re-check it when one of those pins moves.
var nfdPubChartRules = []nfdPubChartRule{
	{component: "network-operator", enablePath: []string{"nfd", "deployNodeFeatureRules"}, chartDefault: true, feature: "pci.device"},
	{component: "gpu-operator", enablePath: []string{"nfd", "nodefeaturerules"}, chartDefault: false, feature: "kernel.loadedmodule"},
	{component: "k8s-nim-operator", enablePath: []string{"nfd", "nodeFeatureRules", "deviceID"}, chartDefault: false, feature: "pci.device"},
	{component: "k8s-nim-operator-ocp", enablePath: []string{"nfd", "nodeFeatureRules", "deviceID"}, chartDefault: false, feature: "pci.device"},
}

var nfdPubManifestFeatureRe = regexp.MustCompile(`(?m)^\s*(?:-\s*)?feature:\s*["']?([A-Za-z0-9_.]+)["']?\s*$`)

// TestNFDFeaturePublishingContract resolves every shipped leaf and checks, on
// merged effective values:
//
//  1. A recipe with network-operator has exactly one nfd-worker sources.custom
//     rule producing pci-15b3.present="true" (the NicClusterPolicy
//     nodeAffinity and the RDMA readiness gate cohort), matching vendor 15b3
//     and device 101c/101e on AKS, vendor 15b3 and class 0200/0207 elsewhere.
//  2. A recipe without network-operator has no such rule, whatever its value:
//     the GPU Operator validator waits for the network-operator MOFED driver
//     on labeled nodes when GPUDirect RDMA is on and useHostMofed is false.
//
// A rule label counts with or without the feature.node.kubernetes.io/ prefix,
// because nfd-master adds that prefix to an un-namespaced label.
//  3. No NodeFeatureRule the recipe renders (chart-shipped or a manifest)
//     reads a feature that noPublishFeatures strips.
func TestNFDFeaturePublishingContract(t *testing.T) {
	ctx := context.Background()
	store, err := buildMetadataStore(ctx, defaultEmbeddedProvider)
	if err != nil {
		t.Fatalf("buildMetadataStore: %v", err)
	}

	leaves := 0
	for name, overlay := range store.Overlays {
		if overlay.Spec.Criteria == nil {
			continue
		}
		leaves++
		criteria := overlay.Spec.Criteria
		t.Run(name, func(t *testing.T) {
			result, err := store.BuildRecipeResult(ctx, criteria)
			if err != nil {
				t.Fatalf("resolve %s: %v", name, err)
			}
			if !slices.Contains(result.Metadata.AppliedOverlays, name) {
				t.Fatalf("%s: criteria resolved to %v, which does not include it", name, result.Metadata.AppliedOverlays)
			}

			hasNFD := nfdPubEnabled(result.ComponentRefs, nfdPubNFD)
			hasNetOp := nfdPubEnabled(result.ComponentRefs, nfdPubNetworkOperator)

			var patterns []string
			var rules []map[string]any
			if hasNFD {
				values, err := result.GetValuesForComponentWithContext(ctx, nfdPubNFD)
				if err != nil {
					t.Fatalf("%s: nfd values: %v", name, err)
				}
				patterns = nfdPubStrings(nfdPubLookup(values, "worker", "config", "core", "noPublishFeatures"))
				rules = nfdPubMellanoxRules(values)
			}

			switch {
			case hasNetOp && !hasNFD:
				t.Fatalf("%s ships network-operator without the nfd component, so nothing labels %s", name, nfdPubMellanoxLabel)
			case hasNetOp:
				if len(rules) != 1 {
					t.Fatalf("%s ships network-operator: want exactly one nfd-worker sources.custom rule labeling %s (or %s), got %d; "+
						"set the nfd componentRef valuesFile to components/nfd/values-nvidia-nics.yaml (values-nvidia-nics-aks.yaml on AKS)",
						name, nfdPubMellanoxLabel, nfdPubMellanoxBare, len(rules))
				}
				for _, problem := range nfdPubNICRuleProblems(rules[0], criteria.Service == CriteriaServiceAKS) {
					t.Errorf("%s: %s", name, problem)
				}
			default:
				if len(rules) != 0 {
					t.Errorf("%s has no network-operator but nfd-worker labels %s (%d rules): the GPU Operator validator would wait "+
						"for a MOFED driver nothing installs", name, nfdPubMellanoxLabel, len(rules))
				}
			}

			if len(patterns) == 0 {
				return
			}
			for _, cr := range nfdPubChartRules {
				if !nfdPubEnabled(result.ComponentRefs, cr.component) {
					continue
				}
				values, err := result.GetValuesForComponentWithContext(ctx, cr.component)
				if err != nil {
					t.Fatalf("%s: %s values: %v", name, cr.component, err)
				}
				enabled := cr.chartDefault
				if v, ok := nfdPubLookup(values, cr.enablePath...).(bool); ok {
					enabled = v
				}
				if enabled && nfdPubStripped(cr.feature, patterns) {
					t.Errorf("%s: %s renders a NodeFeatureRule (%s) reading %s, which nfd.worker.config.core.noPublishFeatures strips",
						name, cr.component, strings.Join(cr.enablePath, "."), cr.feature)
				}
			}
			for _, ref := range result.ComponentRefs {
				if !ref.IsEnabled() {
					continue
				}
				for _, p := range slices.Concat(ref.PreManifestFiles, ref.ManifestFiles) {
					content, err := GetManifestContentWithContext(ctx, nil, p)
					if err != nil {
						t.Fatalf("%s: read %s: %v", name, p, err)
					}
					for _, f := range nfdPubManifestRuleFeatures(string(content)) {
						if nfdPubStripped(f, patterns) {
							t.Errorf("%s: manifest %s has a NodeFeatureRule reading %s, which noPublishFeatures strips", name, p, f)
						}
					}
				}
			}
		})
	}
	if leaves == 0 {
		t.Fatal("no leaf overlays found; the overlay walker is broken")
	}
}

// TestNFDPubStripped pins the pattern semantics to nfd-worker v0.19.0
// patternMatches: exact key, or prefix when the pattern ends in "*".
func TestNFDPubStripped(t *testing.T) {
	tests := []struct {
		name     string
		feature  string
		patterns []string
		want     bool
	}{
		{"exact match", "kernel.config", []string{"kernel.config"}, true},
		{"exact pattern does not prefix-match", "kernel.configs", []string{"kernel.config"}, false},
		{"prefix pattern", "pci.device", []string{"pci.*"}, true},
		{"sibling key kept", "kernel.loadedmodule", []string{"kernel.config", "kernel.enabledmodule"}, false},
		{"no patterns", "pci.device", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nfdPubStripped(tt.feature, tt.patterns); got != tt.want {
				t.Errorf("nfdPubStripped(%q, %v) = %v, want %v", tt.feature, tt.patterns, got, tt.want)
			}
		})
	}
}

func nfdPubEnabled(refs []ComponentRef, name string) bool {
	ref, ok := findComponentRefByName(refs, name)
	return ok && ref.IsEnabled()
}

func nfdPubLookup(m map[string]any, path ...string) any {
	var cur any = m
	for _, k := range path {
		mm, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur = mm[k]
	}
	return cur
}

func nfdPubStrings(v any) []string {
	list, _ := v.([]any)
	out := make([]string, 0, len(list))
	for _, e := range list {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func nfdPubStripped(feature string, patterns []string) bool {
	for _, p := range patterns {
		if prefix, ok := strings.CutSuffix(p, "*"); ok {
			if strings.HasPrefix(feature, prefix) {
				return true
			}
		} else if feature == p {
			return true
		}
	}
	return false
}

func nfdPubMellanoxRules(values map[string]any) []map[string]any {
	var out []map[string]any
	list, _ := nfdPubLookup(values, "worker", "config", "sources", "custom").([]any)
	for _, e := range list {
		rule, ok := e.(map[string]any)
		if !ok {
			continue
		}
		labels, _ := rule["labels"].(map[string]any)
		_, prefixed := labels[nfdPubMellanoxLabel]
		_, bare := labels[nfdPubMellanoxBare]
		if prefixed || bare {
			out = append(out, rule)
		}
	}
	return out
}

// nfdPubNICRuleProblems compares a pci-15b3.present rule against the
// values-nvidia-nics*.yaml contract. On AKS a class match would also label
// the Mellanox Ethernet VFs that Azure accelerated networking attaches, so
// AKS matches the InfiniBand VF device IDs instead.
func nfdPubNICRuleProblems(rule map[string]any, aks bool) []string {
	var problems []string
	labels, _ := rule["labels"].(map[string]any)
	for _, key := range []string{nfdPubMellanoxLabel, nfdPubMellanoxBare} {
		if v, ok := labels[key]; ok && v != "true" {
			problems = append(problems, fmt.Sprintf("rule %v label %s: got %#v, want the string \"true\"", rule["name"], key, v))
		}
	}

	exprs := nfdPubPCIExpressions(rule)
	want := []struct {
		key    string
		values []string
	}{
		{"vendor", []string{"15b3"}},
		{"class", []string{"0200", "0207"}},
	}
	forbidden := "device"
	if aks {
		want[1].key, want[1].values = "device", []string{"101c", "101e"}
		forbidden = "class"
	}
	for _, w := range want {
		op, values, ok := nfdPubExpression(exprs, w.key)
		if !ok {
			problems = append(problems, fmt.Sprintf("rule %v pci.device %s: got no expression, want op \"In\" value %q",
				rule["name"], w.key, w.values))
			continue
		}
		if op != "In" || !slices.Equal(values, w.values) {
			problems = append(problems, fmt.Sprintf("rule %v pci.device %s: got op %q value %q, want op \"In\" value %q",
				rule["name"], w.key, op, values, w.values))
		}
	}
	if _, ok := exprs[forbidden]; ok {
		problems = append(problems, fmt.Sprintf("rule %v pci.device %s: got an expression, want none (aks=%v)",
			rule["name"], forbidden, aks))
	}
	return problems
}

// nfdPubExpression decodes one matchExpressions entry, {op: In, value: [...]}.
// A non-string value element is rendered with its type so it never equals a
// wanted ID.
func nfdPubExpression(exprs map[string]any, key string) (string, []string, bool) {
	e, ok := exprs[key].(map[string]any)
	if !ok {
		return "", nil, false
	}
	op, _ := e["op"].(string)
	list, _ := e["value"].([]any)
	values := make([]string, 0, len(list))
	for _, v := range list {
		if s, ok := v.(string); ok {
			values = append(values, s)
		} else {
			values = append(values, fmt.Sprintf("%T(%v)", v, v))
		}
	}
	return op, values, true
}

func nfdPubPCIExpressions(rule map[string]any) map[string]any {
	matchers, _ := rule["matchFeatures"].([]any)
	for _, m := range matchers {
		mm, ok := m.(map[string]any)
		if !ok || mm["feature"] != "pci.device" {
			continue
		}
		exprs, _ := mm["matchExpressions"].(map[string]any)
		return exprs
	}
	return nil
}

func nfdPubManifestRuleFeatures(content string) []string {
	var out []string
	for _, doc := range strings.Split(content, "\n---") {
		if !strings.Contains(doc, "kind: NodeFeatureRule") && !strings.Contains(doc, "kind: NodeFeatureGroup") {
			continue
		}
		for _, m := range nfdPubManifestFeatureRe.FindAllStringSubmatch(doc, -1) {
			out = append(out, m[1])
		}
	}
	return out
}
