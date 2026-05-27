-- move-code.lua
-- Moves labeled code blocks to an appendix section with automatic references
-- This version works with Quarto's processed code blocks

local code_blocks = {}

-- This runs on CodeBlock elements in the AST
function CodeBlock(el)
  -- For Quarto code blocks, check if there's an lst-cap attribute (which indicates a listing)
  local has_caption = el.attributes and el.attributes["lst-cap"]
  
  if has_caption then
    -- Generate a label from the lst-cap or use default
    local caption = el.attributes["lst-cap"] or "Code listing"
    -- Create an identifier from the caption if not present
    local identifier = el.identifier
    if not identifier or identifier == "" then
      -- Use "lst-code-" plus a hash/counter
      identifier = "lst-code-" .. tostring(math.floor(math.random() * 100000))
    end
    
    -- Store for later insertion
    table.insert(code_blocks, {
      text = el.text,
      caption = caption,
      identifier = identifier,
      classes = el.classes,
      attributes = el.attributes
    })
    
    -- Replace with a reference to the listing in appendix
    return pandoc.Para({
      pandoc.Str("See code "),
      pandoc.RawInline("latex", "\\ref{" .. identifier .. ""),
      -- pandoc.Str("Listing " .. identifier:gsub("lst%-code%-", "")),
      pandoc.RawInline("latex", "}"),
      pandoc.Str(" in Appendix for the implementation.")
    })
  end
  
  -- Return unmodified if not a labeled code block
  return el
end

-- This runs on the entire document
function Pandoc(doc)
  if #code_blocks == 0 then
    return doc
  end
  
  local new_blocks = {}
  local code_inserted = false
  local appendix_index = nil
  
  -- First pass: find the appendix header and collect other blocks
  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" then
      local header_text = pandoc.utils.stringify(block.content)
      
      -- Look for "Appendix: Source code" header
      if string.match(header_text, "Appendix") and string.match(header_text, "[Ss]ource") then
        code_inserted = true
        appendix_index = #new_blocks + 1
      end
    end
    table.insert(new_blocks, block)
  end
  
  -- If appendix found, insert code blocks right after the header
  if code_inserted and appendix_index then
    local code_content = {}
    
    -- Create code blocks with labels for cross-referencing
    for _, item in ipairs(code_blocks) do
      -- Add a label before the listing for hyperref
      -- table.insert(code_content, pandoc.RawBlock("latex", "\\label{" .. item.identifier .. "}"))
      
      -- Create a code block with proper formatting for listings
      table.insert(code_content, pandoc.RawBlock("latex",
        string.format("\\begin{lstlisting}[caption={%s}, language=R, basicstyle=\\small\\ttfamily, breaklines=true, frame=single, numbers=left, numberstyle=\\tiny\\color{gray}, label=%s]",
          item.caption, item.identifier)
      ))
      
      table.insert(code_content, pandoc.RawBlock("latex", item.text))
      
      table.insert(code_content, pandoc.RawBlock("latex", "\\end{lstlisting}"))
      table.insert(code_content, pandoc.RawBlock("latex", ""))
    end
    
    -- Insert after appendix header
    for i = #code_content, 1, -1 do
      table.insert(new_blocks, appendix_index + 1, code_content[i])
    end
  else
    -- If no appendix found, create one at the end
    table.insert(new_blocks, pandoc.Header(2, {pandoc.Str("Appendix: Source code")}))
    
    for _, item in ipairs(code_blocks) do
      -- table.insert(new_blocks, pandoc.RawBlock("latex", "\\label{" .. item.identifier .. "}"))
      
      table.insert(new_blocks, pandoc.RawBlock("latex",
        string.format("\\begin{lstlisting}[caption={%s}, language=R, basicstyle=\\small\\ttfamily, breaklines=true, frame=single, numbers=left, numberstyle=\\tiny\\color{gray}, label=%s]",
          item.caption, item.identifier)
      ))
      table.insert(new_blocks, pandoc.RawBlock("latex", item.text))
      table.insert(new_blocks, pandoc.RawBlock("latex", "\\end{lstlisting}"))
      table.insert(new_blocks, pandoc.RawBlock("latex", ""))
    end
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end
