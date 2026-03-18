use std::io::{self, Write};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Candidate {
    value: f64,
    left_tokens: u32,
    left_index: u32,
    right_tokens: u32,
    right_index: u32,
}

fn reconstruct_expression(levels: &[Vec<Candidate>], tokens: u32, index: u32) -> Vec<String> {
    if tokens == 1 {
        return vec!["1".to_string()];
    }
    let candidate = &levels[tokens as usize][index as usize];
    let mut left_code = reconstruct_expression(levels, candidate.left_tokens, candidate.left_index);
    let mut right_code = reconstruct_expression(levels, candidate.right_tokens, candidate.right_index);
    left_code.append(&mut right_code);
    left_code.push("'EML'".to_string());
    left_code
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut target = std::f64::NAN;
    let mut max_tokens = 41;
    let mut tolerance = 1e-10;

    while let Some(arg) = args.next() {
        if arg == "--target" {
            target = args.next().unwrap().parse().unwrap();
        } else if arg == "--max-tokens" {
            max_tokens = args.next().unwrap().parse().unwrap();
        } else if arg == "--tolerance" {
            tolerance = args.next().unwrap().parse().unwrap();
        }
    }

    let mut levels: Vec<Vec<Candidate>> = vec![Vec::new(); (max_tokens + 1) as usize];
    levels[1].push(Candidate {
        value: 1.0,
        left_tokens: 0,
        left_index: 0,
        right_tokens: 0,
        right_index: 0,
    });

    if (1.0 - target).abs() < tolerance && (target - 1.0).abs() < 1e-12 {
        println!("CANDIDATE: ['1']");
        return;
    }

    for tokens in (3..=max_tokens).step_by(2) {
        let mut level_candidates: Vec<Candidate> = (1..tokens).step_by(2).collect::<Vec<_>>().into_par_iter().flat_map(|left_tokens| {
            let right_tokens = tokens - left_tokens - 1;
            let left_level = &levels[left_tokens as usize];
            let right_level = &levels[right_tokens as usize];
            
            let mut local_cands = Vec::new();
            
            for (left_index, left) in left_level.iter().enumerate() {
                let exp_left = left.value.exp();
                if !exp_left.is_finite() { continue; }
                
                for (right_index, right) in right_level.iter().enumerate() {
                    let log_right = right.value.ln();
                    if !log_right.is_finite() { continue; }
                    
                    let value = exp_left - log_right;
                    if value.is_finite() {
                        local_cands.push(Candidate {
                            value,
                            left_tokens,
                            left_index: left_index as u32,
                            right_tokens,
                            right_index: right_index as u32,
                        });
                    }
                }
            }
            local_cands
        }).collect();

        // Par sort
        level_candidates.par_sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut level = Vec::with_capacity(level_candidates.len() / 2);
        let mut last_bits = None;
        for cand in level_candidates {
            let bits = cand.value.to_bits();
            if last_bits != Some(bits) {
                level.push(cand);
                last_bits = Some(bits);
                if (cand.value - target).abs() < tolerance {
                    levels[tokens as usize] = level.clone(); // temporary
                    let code = reconstruct_expression(&levels, tokens, (level.len() - 1) as u32);
                    let format_code = format!("[{}]", code.join(", "));
                    println!("CANDIDATE: {}", format_code);
                    let _ = io::stdout().flush();
                }
            }
        }
        println!("DEBUG: Level {} generated unique: {}", tokens, level.len());
        let _ = io::stdout().flush();
        levels[tokens as usize] = level;
    }
}
