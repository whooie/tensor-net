#!/usr/bin/nu

$env.config.table.mode = "compact"

def clearall [] { clear; run-external "printf" "'\\e[3J'" }

def main [] { }

# Compile/execute a binary with cargo and run all associated python scripts.
def "main exec" [
  --compute-only (-C) # Only run the executable
  --scripts-only (-S) # Only run the associated scripts
  --pass-stdout (-p) # pass line-delineated STDOUT from the executable as arguments to the scripts
  --clear (-c) # Clear the terminal before doing anything
  --list (-l) # List available target executables
  target?: string # Name of executable
]: nothing -> nothing {
  let execs = (open Cargo.toml | get bin | get name)
  let scriptdirs = (ls -s src | where type == "dir" | get name)
  let targets = (
    $execs | where $it in $scriptdirs
    | append ($scriptdirs | where $it in $execs)
    | uniq
    | sort
  )
  if not (($target == null) or ($target in $targets)) {
    error make { msg: $"invalid target ($target)" }
  }

  if $clear { clearall }
  if $list or ($target == null) {
    echo "Available targets:"
    $targets | each {|t| print $"  ($t)" }
    return
  }
  let exec_out = null
  if not $scripts_only {
    cargo run --release --bin $target
  }
  # let exec_out = if not $scripts_only {
  #   if $pass_stdout {
  #     try {
  #       cargo run --release --bin $target
  #       | complete
  #       | get stdout
  #       | split row "\n"
  #       | where ($it | str length) > 0
  #     } catch {
  #       null
  #     }
  #   } else {
  #     cargo run --release --bin $target
  #     null
  #   }
  # } else {
  #   null
  # }

  if not $compute_only {
    ls $"src/($target)"
    | get name
    | where ($it | str ends-with ".py")
    | sort
    | each {|py|
        print $"(ansi white_bold):: python ($py)(ansi reset)"
        if $exec_out == null {
          python $py
        } else {
          python $py ...$exec_out
        }
      }
  }

  return
}

