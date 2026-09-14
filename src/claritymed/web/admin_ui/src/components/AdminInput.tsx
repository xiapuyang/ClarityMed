import { TextInput, type TextInputProps } from "@mantine/core";
import { forwardRef } from "react";

// Project-wide rule: every <input> needs autoComplete="off" so Chrome's
// "save identity card?" prompt never appears for numeric admin fields.
// Use this component wherever a free-form admin input is added.
export const AdminInput = forwardRef<HTMLInputElement, TextInputProps>(
  (props, ref) => <TextInput ref={ref} autoComplete="off" {...props} />,
);
AdminInput.displayName = "AdminInput";
