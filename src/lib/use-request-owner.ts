import { useEffect, useMemo } from "react";
import { RequestOwner } from "./request-owner";

export function useRequestOwner(baseUrl: string, apiKey: string) {
  const owner = useMemo(() => new RequestOwner(), [baseUrl, apiKey]);
  useEffect(() => {
    owner.activate();
    return () => owner.close();
  }, [owner]);
  return owner;
}
