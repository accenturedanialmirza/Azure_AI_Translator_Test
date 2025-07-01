# Project Flowchart

```mermaid
graph TD
    A[Start] --> B{Read Input CSV};
    B --> C{Detect Language};
    C --> D{Decide Batch Size};
    D --> E{Initialize Translator};
    E --> F{Process Translation (Lazy)};
    F --> G{Split Source Texts into Sentences};
    G --> H{Split Translated Texts into Sentences};
    H --> I{Detect Non-Informative Comments};
    I --> J{Save Translated Data (Parquet)};
    J --> K{Split Sentences into Rows};
    K --> L{Save Split Sentences Data (Parquet)};
    L --> M[End];

    subgraph Data Ingestion
        B
    end

    subgraph Language Detection
        C
    end

    subgraph Translation Process
        D
        E
        F
    end

    subgraph Text Splitting
        G
        H
        K
    end

    subgraph Non-Informative Detection
        I
    end

    subgraph Data Output
        J
        L
    end
```

## How to Render this Flowchart

To view this flowchart as an image, you can use a Mermaid-compatible viewer or tool. Here are a few options:

1.  **VS Code with Mermaid Extension**: If you have VS Code, install the "Markdown Preview Enhanced" or "Mermaid" extension. Open this `project_flowchart.md` file in VS Code, and then open the Markdown preview (usually by clicking the preview icon in the top right of the editor or pressing `Ctrl+Shift+V`).
2.  **Online Mermaid Live Editor**: Copy the content within the ````mermaid ... ```` block and paste it into an online Mermaid Live Editor (e.g., [https://mermaid.live/](https://mermaid.live/)). You can then export the diagram as an SVG or PNG image.
3.  **GitHub**: If you push this Markdown file to a GitHub repository, GitHub will automatically render the Mermaid diagram in the `README.md` or any other Markdown file.