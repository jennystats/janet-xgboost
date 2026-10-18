# project.janet - jpm package manifest for janet-xgboost.
(declare-project
  :name "janet-xgboost"
  :description "In-process xgboost for Janet via core ffi - two ingest routes, bitwise conformance with the official wheel"
  :license "AGPL-3.0"
  :version "0.1.2")

(declare-source
  :source ["jennystats"]
  :dependencies ["https://github.com/jennystats/janet-kit"
                 "https://github.com/jennystats/janet-num"
                 "https://github.com/jennystats/janet-json"])
