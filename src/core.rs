use std::{path::PathBuf, sync::Arc};

use crate::value::Value;

pub struct Span {
    pub file: Arc<PathBuf>,
    pub start: usize,
    pub end: usize,
}

pub struct Node {
    pub span: Span,
    pub value: Value,
}
