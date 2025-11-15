use evolve::reader::Source;
use evolve::runtime::Runtime;

#[test]
fn undefined_symbol_reports_source_span() {
    let runtime = Runtime::new();
    let error = runtime.clone().rep("(foo)", Source::REPL).unwrap_err();

    assert_eq!(error.span, 1..4);
    match error.error {
        evolve::error::Error::RuntimeError(msg) => {
            assert!(msg.contains("Undefined symbol"));
        }
        other => panic!("Expected runtime error, got {:?}", other),
    }
}

#[test]
fn special_form_argument_error_uses_form_span() {
    let runtime = Runtime::new();
    let error = runtime.clone().rep("(if)", Source::REPL).unwrap_err();

    assert_eq!(error.span, 0..4);
    match error.error {
        evolve::error::Error::SyntaxError(_) => {}
        other => panic!("Expected syntax error, got {:?}", other),
    }
}
