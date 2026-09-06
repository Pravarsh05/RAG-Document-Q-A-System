import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { ChunkingStrategy, RetrievalMode, AppConfig } from "@/types/api";

export function useDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: api.listDocuments,
    refetchInterval: (query) => {
      const data = query.state.data;
      const stillProcessing = data?.some((d) => d.status === "processing");
      return stillProcessing ? 2000 : false;
    },
  });
}

export function useIngestDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ file, strategy }: { file: File; strategy: ChunkingStrategy }) =>
      api.ingestDocument(file, strategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useDocumentChunks(documentId: string | null) {
  return useQuery({
    queryKey: ["document-chunks", documentId],
    queryFn: () => (documentId ? api.getDocumentChunks(documentId) : Promise.resolve([])),
    enabled: Boolean(documentId),
  });
}

export function useAskQuestion() {
  return useMutation({
    mutationFn: ({
      question,
      mode,
      filterDocumentId,
      signal,
    }: {
      question: string;
      mode: RetrievalMode;
      filterDocumentId?: string;
      signal?: AbortSignal;
    }) => api.askQuestion(question, mode, filterDocumentId, signal),
  });
}

export function useEvalResults() {
  return useQuery({
    queryKey: ["eval-results"],
    queryFn: api.getEvalResults,
  });
}

export function useRunEval() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.runEval(),
    onSuccess: (data) => {
      queryClient.setQueryData(["eval-results"], data);
    },
  });
}

export function useConfig() {
  return useQuery({
    queryKey: ["config"],
    queryFn: api.getConfig,
  });
}

export function useUpdateConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (cfg: Partial<AppConfig>) => api.updateConfig(cfg),
    onSuccess: (data) => {
      queryClient.setQueryData(["config"], data);
    },
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
    refetchInterval: 10000,
  });
}
