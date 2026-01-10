"""Instrumentation to intercept all decode operations using sys.settrace"""
import sys
import traceback

# Track decode calls
decode_calls = []

def trace_calls(frame, event, arg):
    """Trace function to intercept decode calls"""
    if event == 'call':
        code = frame.f_code
        filename = code.co_filename
        func_name = code.co_name
        
        # Check if this is a decode call by inspecting the code
        if 'decode' in func_name.lower() or 'decode' in str(code.co_names):
            # This might be a decode call, log it
            pass
    
    elif event == 'line':
        # Check if current line contains .decode(
        code = frame.f_code
        filename = code.co_filename
        lineno = frame.f_lineno
        
        # Get the source line if possible
        try:
            import linecache
            line = linecache.getline(filename, lineno)
            if '.decode(' in line and 'utf-8' in line.lower():
                # This is likely a decode call
                stack = traceback.extract_stack(frame)
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"POTENTIAL DECODE CALL DETECTED:", file=sys.stderr)
                print(f"  File: {filename}", file=sys.stderr)
                print(f"  Line: {lineno}", file=sys.stderr)
                print(f"  Code: {line.strip()}", file=sys.stderr)
                print(f"  Stack:", file=sys.stderr)
                for s in stack[-3:]:
                    print(f"    {s.filename}:{s.lineno} in {s.name}", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
        except:
            pass
    
    elif event == 'exception':
        # Exception occurred - check if it's UnicodeDecodeError
        exc_type, exc_value, exc_traceback = arg
        if exc_type == UnicodeDecodeError:
            frame = exc_traceback.tb_frame
            while frame:
                code = frame.f_code
                filename = code.co_filename
                lineno = frame.f_lineno
                
                print(f"\n{'#'*80}", file=sys.stderr)
                print(f"UNICODE DECODE ERROR DETECTED:", file=sys.stderr)
                print(f"  File: {filename}", file=sys.stderr)
                print(f"  Line: {lineno}", file=sys.stderr)
                print(f"  Error: {exc_value}", file=sys.stderr)
                print(f"  Full traceback:", file=sys.stderr)
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
                print(f"{'#'*80}\n", file=sys.stderr)
                break
            frame = frame.f_back
    
    return trace_calls

# Install trace
sys.settrace(trace_calls)
print("Decode instrumentation installed (using sys.settrace)", file=sys.stderr)

